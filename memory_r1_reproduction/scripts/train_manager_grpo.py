"""Algorithm 5: train the Memory Manager with a fixed Answer Agent.

Primary data path: Algorithm 1 tuples (build_manager_training_data.py output).
Each tuple is (dialogue_turns, temporal_memory_bank, current_turn,
linked_questions). Every turn carries its teacher-extracted facts
(LLMExtract, Algorithm 5 line 7, precomputed in the data build). Per tuple
the manager starts from an empty bank (M <- {}) and replays every turn of the
tuple's dialogue window — retrieve with the turn's facts -> manager op ->
apply; the frozen Answer Agent then answers the linked questions with the
resulting bank — the exact-match reward on its answers is the training
signal. The teacher-built temporal bank is consumed by the Answer Agent data
builder (Algorithm 2), not by this loop.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_retriever import build_answer_input
from memfactory.modules.memory_updater import build_manager_input
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
    return [
        {
            "dialogue_id": row["dialogue_id"],
            "turn_index": row["turn_index"],
            "participants": row["participants"],
            "temporal_memory_bank": row["temporal_memory_bank"],
            "dialogue_turns": row["dialogue_turns"],
            "current_turn": row["current_turn"],
            "linked_questions": row["linked_questions"],
        }
        for row in rows
    ]


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
    def __init__(self, model_path: str, device: str, trainable: bool):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype).to(self.device)
        self.model.train(trainable)
        self.trainable = trainable

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> tuple[str, torch.Tensor]:
        """Sample a completion; returns (decoded text, raw sampled token ids).

        The token ids are the exact sequence the policy sampled and are kept
        so log_probability scores the actual rollout action instead of a
        re-tokenized reconstruction of the decoded text.
        """
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
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
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        input_ids = torch.cat([prompt_ids.to(self.device), completion_ids.to(self.device)]).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        start = max(prompt_ids.numel() - 1, 0)
        completion_logits = logits[:, start : start + completion_ids.numel(), :]
        completion_labels = labels[:, start : start + completion_ids.numel()]
        return F.log_softmax(completion_logits, dim=-1).gather(2, completion_labels.unsqueeze(-1)).squeeze(-1)



def answer_with_model(
    answer_model: TextModel,
    question: str,
    memories: list[dict[str, Any]],
    participants: list[str],
    max_new_tokens: int,
    temperature: float,
) -> str:
    memories_by_speaker = {
        participant: [memory for memory in memories if memory["speaker"] == participant]
        for participant in participants
    }
    prompt = build_answer_input(question, memories_by_speaker)
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

    action_losses = []
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
        advantages = (local_rewards - local_rewards.mean()) / (
            local_rewards.std(unbiased=False) + 1e-6
        )
        for (trajectory, _), advantage in zip(active, advantages):
            action = next(
                action for action in trajectory["actions"]
                if action["turn_position"] == position
            )
            prompt = action["prompt"]
            completion_ids = action["completion_ids"]
            old_token_log_probs = action["old_token_log_probs"]
            token_log_probs = manager.log_probability(prompt, completion_ids)  # [L']
            if token_log_probs.numel() == 0:
                continue
            if args.algorithm == "ppo":
                ratio = torch.exp(token_log_probs.sum() - old_token_log_probs.sum())
                clipped = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
                action_losses.append(-torch.minimum(ratio * advantage, clipped * advantage))
            else:
                # GRPO: per-token clipped importance ratio, averaged over the
                # action tokens (same objective shape as the Answer Agent trainer).
                ratio = torch.exp(token_log_probs - old_token_log_probs)
                clipped = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
                per_token = -torch.minimum(ratio * advantage, clipped * advantage)
                action_losses.append(per_token.mean())
            if reference is not None:
                with torch.no_grad():
                    reference_log_probs = reference.log_probability(prompt, completion_ids)
                if reference_log_probs.numel() == 0:
                    continue
                log_ratio = reference_log_probs - token_log_probs
                # R1-style KL penalty k3(u) = e^u - u - 1 (u = log ref - log act), >= 0
                action_losses[-1] = action_losses[-1] + args.beta * (log_ratio.exp() - 1 - log_ratio).mean()
    if not action_losses:
        return float("nan")
    loss = torch.stack(action_losses).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(manager.model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


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
    differences provide turn-level credit in addition to final QA reward.
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
                query = " ".join(fact["text"] for fact in facts)
                old_memory = retrieve(query, memory, args.manager_top_k)
                prompt = build_manager_input(old_memory, facts)
                completion, completion_ids = manager.generate(prompt, args.max_new_tokens, args.temperature)
                with torch.no_grad():
                    old_token_log_probs = manager.log_probability(prompt, completion_ids).detach()
                actions.append(
                    {
                        "prompt": prompt,
                        "completion_ids": completion_ids,
                        "old_token_log_probs": old_token_log_probs,
                        "turn_position": local_index,
                    }
                )
                try:
                    decisions = parse_manager_output(completion)
                    apply_decisions(memory, decisions, item["dialogue_id"], turn)
                    memory_states.append(copy.deepcopy(memory))
                except (json.JSONDecodeError, KeyError, TypeError):
                    valid = False
                    break
            trajectories.append(
                {
                    "memory": memory,
                    "actions": actions,
                    "memory_states": memory_states,
                    "valid": valid,
                }
            )

        rewards: list[list[float]] = []
        for trajectory in trajectories:
            if not trajectory["valid"] or not trajectory["actions"]:
                rewards.append([0.0] * len(trajectory["actions"]))
                continue

            turn_rewards = [0.0] * len(trajectory["actions"])
            previous_potentials = [0.0] * len(questions)
            seen_turn_ids: set[str] = set()
            for action_index, (state, action) in enumerate(zip(trajectory["memory_states"], trajectory["actions"])):
                seen_turn_ids.update(
                    turn["dia_id"]
                    for turn in turns[: action["turn_position"] + 1]
                )
                current_potentials = [
                    evidence_potential(
                        question,
                        state,
                        args.answer_top_k,
                        item["participants"],
                        seen_turn_ids & evidence_ids(question),
                    )
                    for question in questions
                ]
                dense_delta = sum(
                    args.gamma * current - previous
                    for current, previous in zip(current_potentials, previous_potentials)
                )
                if questions:
                    turn_rewards[action_index] += (
                        args.evidence_weight * dense_delta / len(questions)
                    )
                previous_potentials = current_potentials

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
        )


def save_checkpoint(manager: TextModel, output_dir: Path, epoch: int) -> None:
    directory = output_dir / f"epoch_{epoch + 1}"
    directory.mkdir(parents=True, exist_ok=True)
    manager.model.save_pretrained(directory)
    manager.tokenizer.save_pretrained(directory)
    print(f"saved Memory Manager checkpoint to {directory}")


def train(args: argparse.Namespace) -> None:
    manager = TextModel(args.manager_model, args.device, trainable=True)
    reference = TextModel(args.manager_model, args.device, trainable=False) if args.beta else None
    answer = TextModel(args.answer_model, args.device, trainable=False)
    for parameter in answer.model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(manager.model.parameters(), lr=args.learning_rate)
    reward_fn = exact_match_reward if args.reward == "em" else answer_reward

    rows = read_rows(args.data_path)
    tuples = normalize_tuples(rows)
    for item in tuples:
        for turn in item["dialogue_turns"]:
            if not isinstance(turn.get("facts"), list):
                raise SystemExit(
                    f"Turn without extracted facts ({item['dialogue_id']} "
                    f"turn {turn.get('turn_index', '?')}): data must be rebuilt "
                    "with build_manager_training_data.py (LLMExtract step)."
                )
    for epoch in range(args.epochs):
        train_from_tuples(args, manager, reference, answer, optimizer, reward_fn, tuples)
        save_checkpoint(manager, Path(args.output_dir), epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 5: train Memory-R1 Memory Manager.")
    parser.add_argument("--data-path", required=True, help="Algorithm 1 tuples (build_manager_training_data.py output)")
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--output-dir", default="output/memory_r1_manager")
    parser.add_argument("--algorithm", choices=["grpo", "ppo"], default="grpo")
    parser.add_argument("--reward", choices=["em", "f1"], default="em", help="Reward on the frozen answer agent's output (paper uses exact match)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--answer-max-new-tokens", type=int, default=256)
    parser.add_argument("--answer-temperature", type=float, default=1.0)
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount used for dense reward and return-to-go.",
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
