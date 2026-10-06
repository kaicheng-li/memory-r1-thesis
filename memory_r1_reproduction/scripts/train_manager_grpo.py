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
    shard_rows,
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
    another gets short tuples.  The ranks then meet at the per-instance
    gradient all-reduce and the faster ranks sit in NCCL long enough for the
    watchdog to report a false-looking hang.  Greedy cost balancing keeps the
    synchronization points close in time.  Padding only duplicates the least
    expensive item so every rank executes the same number of updates.
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
) -> float:
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
    if action_count == 0:
        return float("nan")

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

    loss_value /= action_count
    average_gradients(manager.model, num_processes)
    torch.nn.utils.clip_grad_norm_(trainable_parameters(manager.model), 1.0)
    optimizer.step()
    return loss_value


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------

def train_from_instances(args, manager, answer, optimizer, reward_fn, instances, accelerator) -> None:
    """Algorithm 5 over QA-conditioned Algorithm 1 temporal tuples.

    Per QA instance: M <- {}; replay every turn of the temporal tuple using
    the turn's teacher-extracted facts, then answer that instance's question
    with the frozen Answer Agent. One GRPO/PPO update uses the group of
    num_generations trajectories for that QA instance.
    """
    progress = tqdm(
        instances,
        desc=f"memory-manager-{args.algorithm}",
        disable=not accelerator.is_local_main_process,
    )
    total_instances = len(instances)
    for instance_index, item in enumerate(progress, start=1):
        question = item["question"]
        turns = item["dialogue_turns"]

        progress.set_description(
            f"memory-manager-{args.algorithm} qa={instance_index}/{total_instances}"
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
        loss = apply_policy_update(
            manager,
            optimizer,
            trajectories,
            rewards,
            args,
            accelerator.num_processes,
        )
        if math.isnan(loss):
            continue
        progress.set_postfix(
            stage="done",
            loss=round(loss, 4),
            reward=round(float(torch.tensor(rewards).mean()), 4),
            refresh=True,
        )


def save_checkpoint(manager: TextModel, output_dir: Path, epoch: int) -> None:
    directory = output_dir / f"epoch_{epoch + 1}"
    directory.mkdir(parents=True, exist_ok=True)
    manager.model.save_pretrained(directory)
    manager.tokenizer.save_pretrained(directory)
    print(f"saved Memory Manager checkpoint to {directory}")


def train(args: argparse.Namespace) -> None:
    accelerator = Accelerator()
    args.device = str(accelerator.device)
    set_seed(args.seed)
    initialize_collectives(args.device, accelerator.num_processes)

    manager = TextModel(args.manager_model, args.device, trainable=True)
    answer = TextModel(
        args.answer_model,
        args.device,
        trainable=False,
        adapter_path=args.answer_adapter,
    )
    optimizer = torch.optim.AdamW(trainable_parameters(manager.model), lr=args.learning_rate)
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
            "generated while replaying each tuple"
        )
    instances = expand_qa_instances(tuples)
    instances = shard_manager_instances(
        instances, accelerator.process_index, accelerator.num_processes
    )
    for epoch in range(args.epochs):
        train_from_instances(args, manager, answer, optimizer, reward_fn, instances, accelerator)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_checkpoint(manager, Path(args.output_dir), epoch)
        accelerator.wait_for_everyone()


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 5: train Memory-R1 Memory Manager.")
    parser.add_argument("--data-path", required=True, help="Algorithm 1 tuples (build_manager_training_data.py output)")
    parser.add_argument("--raw-data-path", required=True, help="Raw LoCoMo JSON used to apply the paper split")
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--answer-adapter")
    parser.add_argument("--output-dir", default="output/memory_r1_manager")
    parser.add_argument("--algorithm", choices=["grpo", "ppo"], default="grpo")
    parser.add_argument("--reward", choices=["em", "f1"], default="em", help="Reward on the frozen answer agent's output (paper uses exact match)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
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
    train(args)


if __name__ == "__main__":
    main()
