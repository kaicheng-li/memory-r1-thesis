"""Algorithm 5: train the Memory Manager with a fixed Answer Agent.

Primary data path: Algorithm 1 tuples (build_manager_training_data.py output).
Each Algorithm 1 temporal tuple is (dialogue_turns, temporal_memory_bank,
current_turn, linked_questions). Every turn carries its teacher-extracted facts
(LLMExtract, Algorithm 5 line 7, precomputed in the data build). Algorithm 5
pairs the temporal tuple with each linked (question, answer) pair. For each
such QA-conditioned instance, the manager starts from an empty bank
(M <- {}) and replays every turn of the tuple's dialogue window — retrieve with
the turn's facts -> manager op -> apply. The frozen Answer Agent then answers
that one question with the resulting bank; its exact-match reward is the
training signal. The teacher-built temporal bank is consumed by the Answer
Agent data builder (Algorithm 2), not by this loop.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_updater import build_manager_input
from memfactory.chat import tokenize_chat_prompt
from memfactory.locomo_split import question_keys
from memfactory.lora import (
    average_gradients,
    initialize_collectives,
    load_lora_model,
    sum_cpu_scalars,
    trainable_parameters,
)
from scripts.construct_memory_bank import (
    apply_decisions,
    parse_manager_output,
    top_k,
    top_k_per_speaker,
)


def read_rows(path: str) -> list[dict[str, Any]]:
    """Read a JSON or JSONL dataset into a list of rows."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


# ---------------------------------------------------------------------------
# Algorithm 1 tuple loading
# ---------------------------------------------------------------------------

def is_tuple_data(rows: list[dict[str, Any]]) -> bool:
    first = next((row for row in rows if isinstance(row, dict)), None)
    return bool(first) and ("temporal_memory_bank" in first or "current_turn" in first)


def normalize_tuples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Algorithm 1 rows -> (dialogue_turns, temporal_memory_bank, current_turn, linked_questions)."""
    tuples = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        current_turn = row.get("current_turn")
        if not isinstance(current_turn, dict):
            continue
        bank = row.get("temporal_memory_bank", [])
        dialogue_turns = [
            turn
            for turn in row.get("dialogue_turns", [])
            if isinstance(turn, dict) and str(turn.get("text", "")).strip()
        ]
        tuples.append(
            {
                "dialogue_id": str(row.get("dialogue_id", "dialogue")),
                "participants": row.get("participants", []),
                "turn_index": int(row.get("turn_index", 0)),
                "temporal_memory_bank": bank if isinstance(bank, list) else [],
                "dialogue_turns": dialogue_turns,
                "current_turn": current_turn,
                "linked_questions": [
                    question
                    for question in row.get("linked_questions", [])
                    if isinstance(question, dict) and question.get("question")
                ],
            }
        )
    return tuples


def expand_qa_instances(tuples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match Algorithm 5's one-(q_i, a_i)-pair training loop."""
    return [
        {**item, "question": question}
        for item in tuples
        for question in item["linked_questions"]
    ]


def shard_manager_instances(
    instances: list[dict[str, Any]], process_index: int, num_processes: int
) -> list[dict[str, Any]]:
    """Shard QA instances by replay cost, keeping collective step counts equal.

    A plain round-robin shard can give one rank mostly 25-turn tuples while
    another gets short tuples. Greedy balancing uses replay turns as a static
    cost proxy; generated token lengths can still differ at runtime. Padding
    duplicates the final item so every rank executes the same number of
    updates and therefore the same collective sequence.
    """
    if num_processes == 1:
        return instances

    ranked = sorted(
        enumerate(instances),
        key=lambda pair: (-len(pair[1].get("dialogue_turns", [])), pair[0]),
    )
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_processes)]
    costs = [0] * num_processes
    for _original_index, instance in ranked:
        target = min(range(num_processes), key=lambda index: (costs[index], len(shards[index])))
        shards[target].append(instance)
        costs[target] += max(1, len(instance.get("dialogue_turns", [])))

    target_count = max(len(shard) for shard in shards)
    for shard in shards:
        if not shard:
            shard.append(instances[0])
        while len(shard) < target_count:
            shard.append(shard[-1])
    return shards[process_index]


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def exact_match_reward(prediction: str, gold: str) -> float:
    """Paper's reward: R = EM(y_pred, y_gold)."""
    prediction_tokens = normalize_text(prediction)
    gold_tokens = normalize_text(gold)
    return 1.0 if prediction_tokens and prediction_tokens == gold_tokens else 0.0


def answer_reward(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not gold_tokens:
        return 0.0
    if prediction_tokens == gold_tokens:
        return 1.0
    overlap = len(set(prediction_tokens) & set(gold_tokens))
    precision = overlap / len(prediction_tokens) if prediction_tokens else 0.0
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def extract_answer(text: str) -> str:
    tagged = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if tagged:
        return tagged.group(1).strip()
    marked = re.search(r"\*\*Answer:\*\*\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if marked:
        return marked.group(1).strip()
    answer = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    return answer.group(1).strip() if answer else text.strip()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TextModel:
    def __init__(
        self,
        model_path: str,
        device: str,
        trainable: bool,
        adapter_path: str | None = None,
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        self.model = load_lora_model(model_path, self.device, adapter_path, trainable)
        self.model.train(trainable)
        self.trainable = trainable

    def tokenize_prompt(self, prompt: str, max_prompt_tokens: int):
        return tokenize_chat_prompt(self.tokenizer, prompt, max_prompt_tokens)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        max_prompt_tokens: int,
    ) -> tuple[str, torch.Tensor]:
        """Sample a completion; returns (decoded text, raw sampled token ids).

        The token ids are the exact sequence the policy sampled and are kept
        so log_probability scores the actual rollout action instead of a
        re-tokenized reconstruction of the decoded text.
        """
        inputs = self.tokenize_prompt(prompt, max_prompt_tokens).to(self.device)
        was_training = self.model.training
        was_checkpointing = self.model.is_gradient_checkpointing
        self.model.eval()
        if was_checkpointing:
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        self.model.config.use_cache = False
        if was_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.train(was_training)
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = output[0][prompt_length:]
        eos_positions = (completion_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=False)
        if len(eos_positions):
            completion_ids = completion_ids[: int(eos_positions[0]) + 1]
        completion_ids = completion_ids.clone()
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return text, completion_ids

    def log_probability(
        self,
        prompt: str,
        completion_ids: torch.Tensor,
        max_prompt_tokens: int,
    ) -> torch.Tensor:
        """Token-level log probabilities over the sampled completion, shape [L'].

        Scores the exact token ids returned by generate(), so importance
        ratios refer to the rollout action that was actually sampled.
        """
        if completion_ids.numel() == 0:
            return torch.zeros((0,), device=self.device)
        prompt_ids = self.tokenize_prompt(prompt, max_prompt_tokens)["input_ids"][0]
        input_ids = torch.cat([prompt_ids.to(self.device), completion_ids.to(self.device)]).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        causal_lm = self.model.get_base_model() if hasattr(self.model, "peft_config") else self.model
        hidden_states = causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        start = max(prompt_ids.numel() - 1, 0)
        action_states = hidden_states[:, start : start + completion_ids.numel(), :]
        labels = input_ids[:, start + 1 : start + 1 + completion_ids.numel()]
        logits = causal_lm.lm_head(action_states)
        selected = logits.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        return (selected - torch.logsumexp(logits, dim=-1)).squeeze(0)



def answer_with_model(answer_model: TextModel, question: str, memories: list[dict[str, Any]], max_new_tokens: int,temperature: float) -> str:
    raise RuntimeError("Use answer_with_model_by_speaker.")


def answer_with_model_by_speaker(
    answer_model: TextModel,
    question: str,
    memories: list[dict[str, Any]],
    participants: list[str],
    max_new_tokens: int,
    temperature: float,
    max_prompt_tokens: int,
) -> str:
    from memfactory.modules.memory_retriever import build_answer_input

    memories_by_speaker = {
        participant: [memory for memory in memories if memory["speaker"] == participant]
        for participant in participants
    }
    prompt = build_answer_input(question, memories_by_speaker)
    raw, _ = answer_model.generate(
        prompt,
        max_new_tokens,
        temperature=temperature,
        max_prompt_tokens=max_prompt_tokens,
    )
    return extract_answer(raw)


# ---------------------------------------------------------------------------
# RL update
# ---------------------------------------------------------------------------

def apply_policy_update(
    manager: TextModel,
    optimizer: torch.optim.Optimizer,
    trajectories: list[dict[str, Any]],
    rewards: list[float],
    args: argparse.Namespace,
    num_processes: int,
) -> tuple[float, float, int, bool]:
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=manager.device)
    advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std(unbiased=False) + 1e-6)
    # Do not keep a computation graph for every replayed turn until the end of
    # the QA instance.  A single instance can replay up to 25 turns across G
    # trajectories; retaining all of those graphs is what exhausts 48 GB GPUs.
    action_count = sum(
        1
        for trajectory in trajectories
        for _prompt, completion_ids, *_rest in trajectory["actions"]
        if completion_ids.numel() > 0
    )
    optimizer.zero_grad()
    loss_value = 0.0
    for trajectory, advantage in zip(trajectories, advantages):
        for action in trajectory["actions"]:
            prompt, completion_ids = action[:2]
            token_log_probs = manager.log_probability(
                prompt, completion_ids, args.max_prompt_tokens
            )  # [L']
            if token_log_probs.numel() == 0:
                continue
            # Rollout and update happen back-to-back before the optimizer
            # step, so the rollout policy is exactly the current policy.  The
            # detached current scores are therefore the old-policy scores;
            # doing a second full Transformer forward during rollout is
            # redundant and greatly increases the NCCL straggler window.
            old_token_log_probs = token_log_probs.detach()
            if args.algorithm == "ppo":
                ratio = torch.exp(token_log_probs.sum() - old_token_log_probs.sum())
                clipped = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
                action_loss = -torch.minimum(ratio * advantage, clipped * advantage)
            else:
                # GRPO: per-token clipped importance ratio, averaged over the
                # action tokens (same objective shape as the Answer Agent trainer).
                ratio = torch.exp(token_log_probs - old_token_log_probs)
                clipped = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
                per_token = -torch.minimum(ratio * advantage, clipped * advantage)
                action_loss = per_token.mean()
            if args.beta > 0:
                was_training = manager.model.training
                manager.model.eval()
                with torch.no_grad():
                    with manager.model.disable_adapter():
                        reference_log_probs = manager.log_probability(
                            prompt, completion_ids, args.max_prompt_tokens
                        )
                manager.model.train(was_training)
                if reference_log_probs.numel() == 0:
                    continue
                log_ratio = reference_log_probs - token_log_probs
                # R1-style KL penalty k3(u) = e^u - u - 1 (u = log ref - log act), >= 0
                action_loss = action_loss + args.beta * (log_ratio.exp() - 1 - log_ratio).mean()

            # Backward immediately, scaled to match the mean over all actions.
            # This releases the transformer activations before scoring the next
            # replayed turn while preserving the original objective.
            (action_loss / action_count).backward()
            loss_value += float(action_loss.detach().item())

    if action_count:
        loss_value /= action_count
    # The failure log showed ranks 0/2/3 enqueueing gradient collectives
    # 3838..3859 while rank 1 was still at 3837. Drain local CUDA work, then use
    # this CPU reduction as the pre-NCCL rendezvous for the next update.
    if manager.device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(manager.device))
    global_action_count = int(sum_cpu_scalars([action_count], num_processes)[0])
    if global_action_count == 0:
        return float("nan"), 0.0, 0, False

    average_gradients(manager.model, num_processes)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(manager.model), 1.0)
    optimizer.step()
    local_loss = loss_value if action_count else float("nan")
    return local_loss, float(grad_norm.detach().item()), action_count, True


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------

def train_from_instances(
    args,
    manager,
    answer,
    optimizer,
    reward_fn,
    instances,
    accelerator,
    epoch_number: int,
    final_epoch: int,
) -> None:
    """Algorithm 5 over QA-conditioned Algorithm 1 temporal tuples.

    Per QA instance: M <- {}; replay every turn of the temporal tuple using
    the turn's teacher-extracted facts, then answer that instance's question
    with the frozen Answer Agent. One GRPO/PPO update uses the group of
    num_generations trajectories for that QA instance.
    """
    progress = tqdm(
        instances,
        desc=f"memory-manager-{args.algorithm} epoch={epoch_number}/{final_epoch}",
        disable=not accelerator.is_local_main_process,
    )
    total_instances = len(instances)
    epoch_metrics = {
        "steps": 0,
        "loss_sum": 0.0,
        "loss_actions": 0.0,
        "reward_sum": 0.0,
        "reward_count": 0.0,
        "reward_nonzero": 0.0,
        "signal_groups": 0.0,
        "groups": 0.0,
        "valid_generations": 0.0,
        "generations": 0.0,
        "grad_norm_sum": 0.0,
        "grad_norm_max": 0.0,
        "actions": 0.0,
    }
    for instance_index, item in enumerate(progress, start=1):
        question = item["question"]
        turns = item["dialogue_turns"]

        progress.set_description(
            f"memory-manager-{args.algorithm} epoch={epoch_number}/{final_epoch} "
            f"qa={instance_index}/{total_instances}"
        )
        progress.set_postfix(stage="start", turns=len(turns), refresh=True)

        trajectories = []
        for generation_index in range(args.num_generations):
            memory: list[dict[str, Any]] = []
            actions = []
            valid = True
            for local_index, turn in enumerate(turns):
                facts = [
                    {
                        "speaker": str(fact.get("speaker", turn.get("speaker", ""))),
                        "timestamp": str(fact.get("timestamp", turn.get("timestamp", ""))),
                        "key": str(fact.get("key", "")),
                        "memory_type": str(fact.get("memory_type", "")),
                        "tags": fact.get("tags", []),
                        "text": str(fact.get("text", "")).strip(),
                    }
                    for fact in turn["facts"]
                    if isinstance(fact, dict) and str(fact.get("text", "")).strip()
                ]
                if not facts:
                    continue
                progress.set_postfix(
                    stage="manager",
                    gen=f"{generation_index + 1}/{args.num_generations}",
                    turn=f"{local_index + 1}/{len(turns)}",
                    refresh=True,
                )
                query = " ".join(fact["text"] for fact in facts)
                old_memory = top_k(query, memory, args.manager_top_k)
                prompt = build_manager_input(old_memory, facts)
                completion, completion_ids = manager.generate(
                    prompt,
                    args.max_new_tokens,
                    args.temperature,
                    args.max_prompt_tokens,
                )
                actions.append((prompt, completion_ids))
                try:
                    decisions = parse_manager_output(completion)
                    apply_decisions(
                        memory,
                        decisions,
                        item["dialogue_id"],
                        int(turn.get("turn_index", local_index)),
                        str(turn.get("speaker", "")),
                        str(turn.get("timestamp", "")),
                    )
                except Exception:
                    valid = False
                    break
            trajectories.append({"memory": memory, "actions": actions, "valid": valid})

        rewards = []
        for generation_index, trajectory in enumerate(trajectories, start=1):
            if not trajectory["valid"] or not trajectory["actions"]:
                rewards.append(0.0)
                continue
            progress.set_postfix(
                stage="answer",
                gen=f"{generation_index}/{len(trajectories)}",
                refresh=True,
            )
            retrieved = top_k_per_speaker(
                question["question"],
                trajectory["memory"],
                item["participants"],
                args.answer_top_k_per_speaker,
            )
            prediction = answer_with_model_by_speaker(
                answer,
                question["question"],
                retrieved,
                item["participants"],
                args.answer_max_new_tokens,
                args.answer_temperature,
                args.max_prompt_tokens,
            )
            rewards.append(reward_fn(prediction, question["answer"]))

        progress.set_postfix(stage="update", refresh=True)
        loss, grad_norm, action_count, did_step = apply_policy_update(
            manager,
            optimizer,
            trajectories,
            rewards,
            args,
            accelerator.num_processes,
        )
        local_loss_sum = 0.0 if math.isnan(loss) else loss * action_count
        local_reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
        local_reward_variance = (
            sum((reward - local_reward_mean) ** 2 for reward in rewards) / len(rewards)
            if rewards
            else 0.0
        )
        valid_generations = sum(
            1 for trajectory in trajectories if trajectory["valid"] and trajectory["actions"]
        )
        (
            global_loss_sum,
            global_loss_actions,
            global_reward_sum,
            global_reward_square_sum,
            global_reward_count,
            global_reward_nonzero,
            global_signal_groups,
            global_valid_generations,
            global_groups,
            global_grad_norm_sum,
            global_actions,
            global_steps,
        ) = sum_cpu_scalars(
            [
                local_loss_sum,
                0.0 if math.isnan(loss) else action_count,
                sum(rewards),
                sum(reward * reward for reward in rewards),
                len(rewards),
                sum(1 for reward in rewards if reward > 0),
                float(local_reward_variance > 1e-12),
                valid_generations,
                1.0,
                grad_norm,
                action_count,
                float(did_step),
            ],
            accelerator.num_processes,
        )
        global_loss = (
            global_loss_sum / global_loss_actions
            if global_loss_actions
            else float("nan")
        )
        global_reward = global_reward_sum / global_reward_count if global_reward_count else 0.0
        global_reward_variance = max(
            0.0,
            global_reward_square_sum / global_reward_count - global_reward**2,
        ) if global_reward_count else 0.0
        global_reward_std = math.sqrt(global_reward_variance)
        mean_grad_norm = global_grad_norm_sum / accelerator.num_processes
        learning_rate = float(optimizer.param_groups[0]["lr"])

        if accelerator.is_main_process:
            epoch_metrics["steps"] += float(global_steps > 0)
            epoch_metrics["loss_sum"] += global_loss_sum
            epoch_metrics["loss_actions"] += global_loss_actions
            epoch_metrics["reward_sum"] += global_reward_sum
            epoch_metrics["reward_count"] += global_reward_count
            epoch_metrics["reward_nonzero"] += global_reward_nonzero
            epoch_metrics["signal_groups"] += global_signal_groups
            epoch_metrics["groups"] += global_groups
            epoch_metrics["valid_generations"] += global_valid_generations
            epoch_metrics["generations"] += global_reward_count
            epoch_metrics["grad_norm_sum"] += mean_grad_norm
            epoch_metrics["grad_norm_max"] = max(
                epoch_metrics["grad_norm_max"], mean_grad_norm
            )
            epoch_metrics["actions"] += global_actions

        progress.set_postfix(
            stage="done",
            loss="nan" if math.isnan(global_loss) else f"{global_loss:.4f}",
            reward=f"{global_reward:.4f}",
            reward_std=f"{global_reward_std:.4f}",
            signal=f"{int(global_signal_groups)}/{int(global_groups)}",
            grad_preclip=f"{mean_grad_norm:.3e}",
            lr=f"{learning_rate:.2e}",
            refresh=True,
        )

    if accelerator.is_main_process:
        steps = max(epoch_metrics["steps"], 1.0)
        loss_actions = max(epoch_metrics["loss_actions"], 1.0)
        reward_count = max(epoch_metrics["reward_count"], 1.0)
        print(
            f"Manager epoch {epoch_number} summary: "
            f"updates={int(epoch_metrics['steps'])}, "
            f"loss={epoch_metrics['loss_sum'] / loss_actions:.6f}, "
            f"reward={epoch_metrics['reward_sum'] / reward_count:.6f}, "
            f"nonzero_reward={int(epoch_metrics['reward_nonzero'])}/"
            f"{int(epoch_metrics['reward_count'])}, "
            f"reward_signal_groups={int(epoch_metrics['signal_groups'])}/"
            f"{int(epoch_metrics['groups'])}, "
            f"valid_generations={int(epoch_metrics['valid_generations'])}/"
            f"{int(epoch_metrics['generations'])}, "
            f"grad_norm_preclip_mean={epoch_metrics['grad_norm_sum'] / steps:.6e}, "
            f"grad_norm_preclip_max={epoch_metrics['grad_norm_max']:.6e}, "
            f"lr={float(optimizer.param_groups[0]['lr']):.6e}, "
            f"actions={int(epoch_metrics['actions'])}",
            flush=True,
        )


def save_checkpoint(manager: TextModel, output_dir: Path, epoch_number: int) -> None:
    """Save a 1-based epoch checkpoint without assuming training starts at 0."""
    directory = output_dir / f"epoch_{epoch_number}"
    directory.mkdir(parents=True, exist_ok=True)
    manager.model.save_pretrained(directory)
    manager.tokenizer.save_pretrained(directory)
    print(f"saved Memory Manager checkpoint to {directory}", flush=True)


def train(args: argparse.Namespace) -> None:
    accelerator = Accelerator()
    args.device = str(accelerator.device)
    set_seed(args.seed)
    initialize_collectives(
        args.device, accelerator.num_processes, enable_cpu_sync=True
    )
    if accelerator.is_main_process:
        print(
            "Manager training config: "
            f"world_size={accelerator.num_processes}, optimizer=AdamW, "
            f"lr={args.learning_rate:.6e}, scheduler=none, "
            f"grad_clip_norm=1.0, beta={args.beta}, "
            f"clip_epsilon={args.clip_epsilon}, reward={args.reward}, "
            f"num_generations={args.num_generations}, temperature={args.temperature}, "
            f"manager_max_new_tokens={args.max_new_tokens}, "
            f"answer_max_new_tokens={args.answer_max_new_tokens}, "
            f"max_prompt_tokens={args.max_prompt_tokens}",
            flush=True,
        )

    manager = TextModel(
        args.manager_model,
        args.device,
        trainable=True,
        adapter_path=args.manager_adapter,
    )
    answer = TextModel(
        args.answer_model,
        args.device,
        trainable=False,
        adapter_path=args.answer_adapter,
    )
    optimizer = torch.optim.AdamW(trainable_parameters(manager.model), lr=args.learning_rate)
    if args.manager_adapter and accelerator.is_main_process:
        print(
            f"Continuing Manager from {args.manager_adapter} with a fresh AdamW "
            "state (older checkpoints do not contain optimizer state).",
            flush=True,
        )
    reward_fn = exact_match_reward if args.reward == "em" else answer_reward

    rows = read_rows(args.data_path)
    if not is_tuple_data(rows):
        raise SystemExit(
            "Data must be Algorithm 1 tuples (build_manager_training_data.py output): "
            "rows have to contain 'temporal_memory_bank'/'current_turn'."
        )
    allowed = question_keys(args.raw_data_path, args.split)
    for row in rows:
        dialogue_id = str(row.get("dialogue_id", "dialogue"))
        row["linked_questions"] = [
            question
            for question in row.get("linked_questions", [])
            if (dialogue_id, str(question.get("question_id", ""))) in allowed
        ]
    tuples = normalize_tuples(rows)
    for item in tuples:
        for turn in item["dialogue_turns"]:
            if not isinstance(turn.get("facts"), list):
                raise SystemExit(
                    f"Turn without extracted facts ({item['dialogue_id']} "
                    f"turn {turn.get('turn_index', '?')}): data must be rebuilt "
                    "with build_manager_training_data.py (LLMExtract step)."
                )
    tuples = [item for item in tuples if item["linked_questions"]]
    if accelerator.is_main_process:
        qa_count = sum(len(item["linked_questions"]) for item in tuples)
        print(
            f"LoCoMo {args.split} split: {qa_count} QA rewards across "
            f"{len(tuples)} Algorithm 1 temporal tuples; Manager actions are "
            "generated while replaying each tuple",
            flush=True,
        )
    instances = expand_qa_instances(tuples)
    unique_instance_count = len(instances)
    instances = shard_manager_instances(
        instances, accelerator.process_index, accelerator.num_processes
    )
    if accelerator.is_main_process:
        scheduled_instances = len(instances) * accelerator.num_processes
        print(
            "Manager distributed schedule: "
            f"steps_per_rank={len(instances)}, "
            f"global_slots={scheduled_instances}, "
            f"unique_QA={unique_instance_count}, "
            f"padding_replicas={scheduled_instances - unique_instance_count}",
            flush=True,
        )
    first_epoch = args.start_epoch + 1
    last_epoch = args.start_epoch + args.epochs
    for epoch_number in range(first_epoch, last_epoch + 1):
        train_from_instances(
            args,
            manager,
            answer,
            optimizer,
            reward_fn,
            instances,
            accelerator,
            epoch_number,
            last_epoch,
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(manager, Path(args.output_dir), epoch_number)
        accelerator.wait_for_everyone()


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 5: train Memory-R1 Memory Manager.")
    parser.add_argument("--data-path", required=True, help="Algorithm 1 tuples (build_manager_training_data.py output)")
    parser.add_argument("--raw-data-path", required=True, help="Raw LoCoMo JSON used to apply the paper split")
    parser.add_argument("--manager-model", required=True)
    parser.add_argument(
        "--manager-adapter",
        help="Existing Manager LoRA adapter to continue training from (for example epoch_4).",
    )
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--answer-adapter")
    parser.add_argument("--output-dir", default="output/memory_r1_manager")
    parser.add_argument("--algorithm", choices=["grpo", "ppo"], default="grpo")
    parser.add_argument("--reward", choices=["em", "f1"], default="em", help="Reward on the frozen answer agent's output (paper uses exact match)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="1-based checkpoint epoch already completed; new checkpoints start at this value + 1.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k-per-speaker", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--answer-max-new-tokens", type=int, default=2048)
    parser.add_argument("--answer-temperature", type=float, default=1.0)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.start_epoch < 0:
        parser.error("--start-epoch cannot be negative")
    try:
        train(args)
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
