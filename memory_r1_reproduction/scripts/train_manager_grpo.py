"""Algorithm 5: train the Memory Manager with a fixed Answer Agent.

Primary data path: Algorithm 1 tuples (build_manager_training_data.py output).
Each tuple is (dialogue_turns, current_turn, linked_questions), plus an
optional unused temporal_memory_bank field for data compatibility. Every
turn carries its teacher-extracted facts (LLMExtract, Algorithm 5 line 7).
Per tuple
the manager starts from an empty bank (M <- {}) and replays every turn of the
tuple's dialogue window — retrieve with the turn's facts -> manager op ->
apply; the frozen Answer Agent then answers the linked questions with the
resulting bank — the exact-match reward on its answers is the training
signal. The Answer tuple builder replays the same windows with a frozen
initial Manager, without consuming the optional teacher temporal bank.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_retriever import build_answer_input_from_memories
from memfactory.modules.memory_updater import build_manager_input
from memfactory.lora import load_lora_model, trainable_parameters
from memfactory.chat import DEFAULT_MAX_PROMPT_TOKENS, tokenize_chat_prompt
from memfactory.locomo_split import SPLIT_DIALOGUE_IDS
from memfactory.run_metadata import write_run_metadata
from memfactory.sequential_validation import load_validation_timelines, evaluate_roll_forward
from memfactory.memory_runtime import apply_decisions, parse_manager_output, retrieve, retrieve_per_speaker


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

def normalize_tuples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Algorithm 1 rows -> (dialogue_turns, temporal_memory_bank, current_turn, linked_questions)."""
    normalized = []
    for row in rows:
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        normalized.append({
            "dialogue_id": str(row["dialogue_id"]),
            "split": row.get("split", metadata.get("split")),
            "turn_index": row["turn_index"],
            "participants": row["participants"],
            "temporal_memory_bank": row["temporal_memory_bank"],
            "dialogue_turns": row["dialogue_turns"],
            "current_turn": row["current_turn"],
            "linked_questions": row["linked_questions"],
        })
    return normalized


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


def evidence_ids(question: dict[str, Any]) -> set[str]:
    """Return source turn ids supplied by LoCoMo QA annotations."""
    return set(question["evidence"])


def memory_source_ids(memory: list[dict[str, Any]]) -> set[str]:
    """Collect provenance ids without treating updated text as new evidence."""
    return {source for entry in memory for source in entry["source_turn_ids"]}


def potential_shaping_rewards(
    potentials: list[float], gamma: float, terminal_potential: str = "observed",
    initial_potential: float = 0.0,
) -> list[float]:
    """Potential differences; observed terminal coverage is an auxiliary goal.

    With a zero terminal potential the discounted sum is -initial_potential.
    That boundary condition alone does not guarantee better GRPO credit.
    """
    if terminal_potential not in {"observed", "zero"}:
        raise ValueError("terminal_potential must be observed or zero")
    values = list(potentials)
    if values and terminal_potential == "zero":
        values[-1] = 0.0
    previous = initial_potential
    rewards = []
    for current in values:
        rewards.append(gamma * current - previous)
        previous = current
    return rewards


def evidence_potential(
    question: dict[str, Any],
    memory: list[dict[str, Any]],
    answer_top_k: int,
    participants: list[str],
    available_evidence: set[str],
) -> float:
    """Measure how much annotated evidence is stored and retrievable.

    This is training-only privileged information. The Answer Agent still sees
    only the retrieved memory, never the original dialogue or evidence labels.
    """
    all_evidence = evidence_ids(question)
    if not all_evidence:
        return 0.0

    stored = memory_source_ids(memory)
    memory_coverage = len(available_evidence & stored) / len(all_evidence)

    retrieved = retrieve_per_speaker(question["question"], memory, participants, answer_top_k)
    retrieved_sources = memory_source_ids(retrieved)
    retrieval_coverage = len(available_evidence & retrieved_sources) / len(all_evidence)
    return 0.5 * memory_coverage + 0.5 * retrieval_coverage


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TextModel:
    def __init__(self, model_path: str, device: str, trainable: bool, adapter_path: str | None = None,
                 max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = load_lora_model(model_path, self.device, adapter_path, trainable)
        self.model.train(trainable)
        self.trainable = trainable
        self.max_prompt_tokens = max_prompt_tokens

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> tuple[str, torch.Tensor]:
        """Sample a completion; returns (decoded text, raw sampled token ids).

        The token ids are the exact sequence the policy sampled and are kept
        so log_probability scores the actual rollout action instead of a
        re-tokenized reconstruction of the decoded text.
        """
        inputs = tokenize_chat_prompt(self.tokenizer, prompt, self.max_prompt_tokens).to(self.device)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_k=0,
                    top_p=1.0,
                    typical_p=1.0,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        finally:
            self.model.train(was_training)
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = output[0][prompt_length:].clone()
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        return text, completion_ids

    def log_probability(self, prompt: str, completion_ids: torch.Tensor) -> torch.Tensor:
        """Token-level log probabilities over the sampled completion, shape [L'].

        Scores the exact token ids returned by generate(), so importance
        ratios refer to the rollout action that was actually sampled.
        """
        if completion_ids.numel() == 0:
            return torch.zeros((0,), device=self.device)
        prompt_ids = tokenize_chat_prompt(self.tokenizer, prompt, self.max_prompt_tokens)["input_ids"][0]
        input_ids = torch.cat([prompt_ids.to(self.device), completion_ids.to(self.device)]).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        was_training = self.model.training
        self.model.eval()
        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]
            start = max(prompt_ids.numel() - 1, 0)
            completion_logits = logits[:, start : start + completion_ids.numel(), :]
            completion_labels = labels[:, start : start + completion_ids.numel()]
            return F.log_softmax(completion_logits, dim=-1).gather(2, completion_labels.unsqueeze(-1)).squeeze(-1)
        finally:
            self.model.train(was_training)



def answer_with_model(
    answer_model: TextModel,
    question: str,
    memories: list[dict[str, Any]],
    participants: list[str],
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = build_answer_input_from_memories(question, memories, participants)
    raw, _ = answer_model.generate(prompt, max_new_tokens, temperature=temperature)
    return raw.rsplit("Answer:", 1)[-1].strip()


# ---------------------------------------------------------------------------
# RL update
# ---------------------------------------------------------------------------

def apply_policy_update(
    manager: TextModel,
    reference: TextModel | None,
    optimizer: torch.optim.Optimizer,
    trajectories: list[dict[str, Any]],
    rewards: list[list[float]],
    args: argparse.Namespace,
) -> float:
    returns = []
    for trajectory_rewards in rewards:
        trajectory_returns = [0.0] * len(trajectory_rewards)
        running_return = 0.0
        for index in reversed(range(len(trajectory_rewards))):
            running_return = trajectory_rewards[index] + args.gamma * running_return
            trajectory_returns[index] = running_return
        returns.append(trajectory_returns)

    eligible_actions = []
    action_positions = sorted({
        action["turn_position"]
        for trajectory in trajectories
        for action in trajectory["actions"]
    })
    for position in action_positions:
        active = []
        for row_index, trajectory in enumerate(trajectories):
            for action_index, action in enumerate(trajectory["actions"]):
                if action["turn_position"] == position:
                    active.append((trajectory, returns[row_index][action_index]))
                    break
        if not active:
            continue
        local_rewards = torch.tensor(
            [reward for _, reward in active],
            dtype=torch.float32,
            device=manager.device,
        )
        reward_std = local_rewards.std(unbiased=False)
        variance_epsilon = getattr(args, "variance_epsilon", 1e-6)
        # Chosen convention: skip both the group policy term and standalone
        # reference regularization when there is no relative reward signal.
        if len(active) < 2 or float(reward_std.item()) <= variance_epsilon:
            continue
        advantages = (local_rewards - local_rewards.mean()) / (reward_std + 1e-6)
        for (trajectory, _), advantage in zip(active, advantages):
            action = next(
                action for action in trajectory["actions"]
                if action["turn_position"] == position
            )
            if action["completion_ids"].numel() > 0:
                eligible_actions.append((action, advantage))
    if not eligible_actions:
        return float("nan")

    optimizer.zero_grad()
    loss_sum = 0.0
    for action, advantage in eligible_actions:
        prompt = action["prompt"]
        completion_ids = action["completion_ids"]
        old_token_log_probs = action["old_token_log_probs"]
        token_log_probs = manager.log_probability(prompt, completion_ids)  # [L']
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
        if reference is not None:
            with torch.no_grad():
                reference_log_probs = reference.log_probability(prompt, completion_ids)
            if reference_log_probs.numel() > 0:
                log_ratio = reference_log_probs - token_log_probs
                # R1-style KL penalty k3(u) = e^u - u - 1, u = log ref - log act.
                action_loss = action_loss + args.beta * (
                    log_ratio.exp() - 1 - log_ratio
                ).mean()
        # Backpropagate now, rather than retaining every long action graph
        # until the end of a sequential rollout group.
        (action_loss / len(eligible_actions)).backward()
        loss_sum += float(action_loss.detach().item())
    torch.nn.utils.clip_grad_norm_(trainable_parameters(manager.model), 1.0)
    optimizer.step()
    return loss_sum / len(eligible_actions)


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------

def train_from_tuples(args, manager, reference, answer, optimizer, reward_fn, tuples) -> None:
    """Algorithm 5 over Algorithm 1 tuples.

    Per tuple: M <- {}; replay every turn of the tuple's dialogue window
    using the turn's teacher-extracted facts (LLMExtract) — retrieve with the
    facts -> manager op -> apply; then answer the linked questions with the
    frozen Answer Agent over the resulting bank. One GRPO/PPO update per
    tuple over its num_generations trajectories. Evidence-aware potential
    differences are added to final QA reward. Their effect on credit depends
    on the terminal boundary and the return/advantage estimator.
    """
    progress = tqdm(tuples, desc=f"memory-manager-{args.algorithm}")
    for item in progress:
        questions = item["linked_questions"]
        if not questions:
            continue
        turns = item["dialogue_turns"]

        trajectories = []
        for _ in range(args.num_generations):
            memory: list[dict[str, Any]] = []
            actions = []
            memory_states = []
            invalid_actions = 0
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
                query = " ".join(fact["text"] for fact in facts)
                old_memory = retrieve(query, memory, args.manager_top_k)
                prompt = build_manager_input(old_memory, facts)
                completion, completion_ids = manager.generate(prompt, args.max_new_tokens, args.temperature)
                with torch.no_grad():
                    old_token_log_probs = manager.log_probability(prompt, completion_ids).detach()
                action = {
                    "prompt": prompt,
                    "completion_ids": completion_ids,
                    "old_token_log_probs": old_token_log_probs,
                    "turn_position": local_index,
                    "valid": True,
                }
                actions.append(action)
                try:
                    decisions = parse_manager_output(completion)
                    apply_decisions(
                        memory,
                        decisions,
                        item["dialogue_id"],
                        turn,
                        visible_memory_ids={entry["id"] for entry in old_memory},
                    )
                except (ValueError, KeyError, TypeError):
                    # Validation is atomic.  Keep the unchanged bank, then
                    # continue so earlier valid actions keep their signal.
                    action["valid"] = False
                    invalid_actions += 1
                memory_states.append(copy.deepcopy(memory))
            trajectories.append(
                {
                    "memory": memory,
                    "actions": actions,
                    "memory_states": memory_states,
                    "invalid_actions": invalid_actions,
                }
            )

        rewards: list[list[float]] = []
        for trajectory in trajectories:
            if not trajectory["actions"]:
                rewards.append([0.0] * len(trajectory["actions"]))
                continue

            turn_rewards = [0.0] * len(trajectory["actions"])
            for action_index, action in enumerate(trajectory["actions"]):
                if not action["valid"]:
                    turn_rewards[action_index] -= args.invalid_action_penalty
            state_potentials = []
            seen_turn_ids: set[str] = set()
            for action_index, (state, action) in enumerate(zip(trajectory["memory_states"], trajectory["actions"])):
                seen_turn_ids.update(
                    turn["dia_id"]
                    for turn in turns[: action["turn_position"] + 1]
                )
                next_potentials = [
                    evidence_potential(
                        question,
                        state,
                        args.answer_top_k,
                        item["participants"],
                        seen_turn_ids & evidence_ids(question),
                    )
                    for question in questions
                ]
                state_potentials.append(sum(next_potentials) / len(questions))
            shaping = potential_shaping_rewards(
                state_potentials, args.gamma, getattr(args, "terminal_potential", "observed")
            )
            for action_index, delta in enumerate(shaping):
                turn_rewards[action_index] += args.evidence_weight * delta

            question_rewards = []
            for question in questions:
                retrieved = retrieve_per_speaker(
                    question["question"],
                    trajectory["memory"],
                    item["participants"],
                    args.answer_top_k,
                )
                prediction = answer_with_model(
                    answer,
                    question["question"],
                    retrieved,
                    item["participants"],
                    args.answer_max_new_tokens,
                    args.answer_temperature,
                )
                question_rewards.append(reward_fn(prediction, question["answer"]))
            turn_rewards[-1] += sum(question_rewards) / len(question_rewards)
            rewards.append(turn_rewards)

        loss = apply_policy_update(manager, reference, optimizer, trajectories, rewards, args)
        if math.isnan(loss):
            continue
        flat_rewards = [value for row in rewards for value in row]
        progress.set_postfix(
            loss=loss,
            reward=float(torch.tensor(flat_rewards).mean()) if flat_rewards else 0.0,
            invalid=sum(row["invalid_actions"] for row in trajectories),
        )


def save_checkpoint(manager: TextModel, output_dir: Path, epoch: int) -> None:
    directory = output_dir / f"epoch_{epoch + 1}"
    directory.mkdir(parents=True, exist_ok=True)
    manager.model.save_pretrained(directory)
    manager.tokenizer.save_pretrained(directory)
    print(f"saved Memory Manager checkpoint to {directory}")


def train(args: argparse.Namespace) -> None:
    if args.temperature != 1.0:
        raise ValueError("Manager training requires temperature=1 to match raw policy log probabilities.")
    if args.num_generations < 2:
        raise ValueError("Group-relative training requires at least two generations.")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    set_seed(args.seed)
    validation = load_validation_timelines(args.validation_timelines) if args.validation_timelines else None
    rows = read_rows(args.data_path)
    tuples = normalize_tuples(rows)
    if not tuples:
        raise ValueError("No Manager training tuples were loaded.")
    if not getattr(args, "allow_nontrain_dialogues", False):
        permitted = set(SPLIT_DIALOGUE_IDS["train"])
        unexpected = sorted({
            str(item["dialogue_id"]) for item in tuples
            if item["dialogue_id"] not in permitted or item["split"] != "train"
        })
        if unexpected:
            raise ValueError(
                "Manager training data contains non-train LoCoMo dialogues: "
                + ", ".join(unexpected)
            )
    for item in tuples:
        for turn in item["dialogue_turns"]:
            if not isinstance(turn.get("facts"), list):
                raise SystemExit(
                    f"Turn without extracted facts ({item['dialogue_id']} "
                    f"turn {turn.get('turn_index', '?')}): data must be rebuilt "
                    "with build_manager_training_data.py (LLMExtract step)."
                )
    manager = TextModel(args.manager_model, args.device, trainable=True, max_prompt_tokens=args.max_prompt_tokens)
    reference = TextModel(args.manager_model, args.device, trainable=False, max_prompt_tokens=args.max_prompt_tokens) if args.beta else None
    answer = TextModel(args.answer_model, args.device, trainable=False, adapter_path=args.answer_adapter,
                       max_prompt_tokens=args.max_prompt_tokens)
    optimizer = torch.optim.AdamW(trainable_parameters(manager.model), lr=args.learning_rate)
    write_run_metadata(args)
    reward_fn = exact_match_reward if args.reward == "em" else answer_reward
    def validate(epoch):
        result = evaluate_roll_forward(
            validation,
            lambda prompt: manager.generate(prompt, args.max_new_tokens, 0.0)[0],
            lambda question, memories, participants: answer_with_model(
                answer, question, memories, participants, args.answer_max_new_tokens, 0.0
            ),
            args, normalize_text, exact_match_reward, answer_reward,
        )
        (Path(args.output_dir) / f"validation_epoch_{epoch}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"validation epoch={epoch}: {result['summary']}")
        return result["summary"]["token_f1_multiset"]

    best_f1 = validate(0) if validation else None
    if validation:
        (Path(args.output_dir) / "best_checkpoint.json").write_text(
            json.dumps({"epoch": 0, "adapter": None, "token_f1_multiset": best_f1}), encoding="utf-8"
        )
    for epoch in range(args.epochs):
        train_from_tuples(args, manager, reference, answer, optimizer, reward_fn, tuples)
        save_checkpoint(manager, Path(args.output_dir), epoch)
        if validation:
            f1 = validate(epoch + 1)
            if f1 > best_f1:
                best_f1 = f1
                (Path(args.output_dir) / "best_checkpoint.json").write_text(json.dumps({
                    "epoch": epoch + 1, "adapter": str(Path(args.output_dir) / f"epoch_{epoch + 1}"),
                    "token_f1_multiset": best_f1,
                }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 5: train Memory-R1 Memory Manager.")
    parser.add_argument("--data-path", required=True, help="Algorithm 1 tuples (build_manager_training_data.py output)")
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--answer-adapter", help="Optional frozen Answer adapter; omit for frozen base Answer model.")
    parser.add_argument("--validation-timelines", help="Cached validation dialogue timelines for full-dialogue checkpoint selection.")
    parser.add_argument(
        "--allow-nontrain-dialogues",
        action="store_true",
        help="Explicitly allow a custom/non-LoCoMo training split; never use for a LoCoMo test claim.",
    )
    parser.add_argument("--output-dir", default="output/memory_r1_manager")
    parser.add_argument("--algorithm", choices=["grpo", "ppo"], default="grpo")
    parser.add_argument("--reward", choices=["em", "f1"], default="em", help="Reward on the frozen answer agent's output (paper uses exact match)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k", type=int, default=30, help="Retrieved memories per speaker, matching Answer tuple construction.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--answer-max-new-tokens", type=int, default=2048)
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument(
        "--invalid-action-penalty",
        type=float,
        default=0.1,
        help="Per-action penalty for malformed or invalid actions; invalid updates are atomic no-ops.",
    )
    parser.add_argument(
        "--variance-epsilon",
        type=float,
        default=1e-6,
        help="Skip a turn-position group when its return standard deviation is at or below this value.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount used for dense reward and return-to-go.",
    )
    parser.add_argument(
        "--terminal-potential", choices=["observed", "zero"], default="observed",
        help="observed adds a terminal coverage goal; zero cancels total shaping at gamma=1 from an empty bank.",
    )
    parser.add_argument(
        "--evidence-weight",
        type=float,
        default=0.5,
        help="Weight of evidence-aware per-turn potential shaping.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
