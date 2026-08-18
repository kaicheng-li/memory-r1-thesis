"""Algorithm 5: train the Memory Manager with a fixed Answer Agent.

Primary data path: Algorithm 1 tuples (build_manager_training_data.py output).
Each tuple is (dialogue_turns, temporal_memory_bank, current_turn,
linked_questions). Per tuple the manager starts from an empty bank (M <- {})
and replays every turn of the tuple's dialogue window; the frozen Answer Agent
then answers the linked questions with the resulting bank — the exact-match
reward on its answers is the training signal. The teacher-built temporal bank
is consumed by the Answer Agent data builder (Algorithm 2), not by this loop.

Fallback data path: raw LoCoMo JSON, in which case the whole dialogue is the
tuple and is rolled out from an empty bank the same way.
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
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_retriever import build_answer_input
from memfactory.modules.memory_updater import build_manager_input


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
                "turn_index": int(row.get("turn_index", 0)),
                "temporal_memory_bank": bank if isinstance(bank, list) else [],
                "dialogue_turns": dialogue_turns or [current_turn],
                "current_turn": current_turn,
                "linked_questions": [
                    question
                    for question in row.get("linked_questions", [])
                    if isinstance(question, dict) and question.get("question")
                ],
            }
        )
    return tuples


# ---------------------------------------------------------------------------
# Raw LoCoMo loading (fallback)
# ---------------------------------------------------------------------------

def normalize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    if "turns" in sample:
        return sample
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo conversation must be an object.")

    def session_number(name: str) -> int:
        match = re.search(r"session_(\d+)$", name)
        return int(match.group(1)) if match else 10**9

    turns = []
    sessions = sorted(
        (name for name, value in conversation.items() if re.fullmatch(r"session_\d+", name) and isinstance(value, list)),
        key=session_number,
    )
    evidence_to_index = {}
    for session_name in sessions:
        for local_index, raw_turn in enumerate(conversation[session_name]):
            if not isinstance(raw_turn, dict):
                continue
            turn_index = len(turns)
            dia_id = str(raw_turn.get("dia_id", f"D{session_number(session_name)}:{local_index}"))
            turns.append(
                {
                    "speaker": str(raw_turn.get("speaker", "")),
                    "timestamp": str(raw_turn.get("timestamp", conversation.get(f"{session_name}_date_time", ""))),
                    "text": str(raw_turn.get("text", "")),
                    "dia_id": dia_id,
                }
            )
            evidence_to_index[dia_id] = turn_index

    questions = []
    for question_index, raw_question in enumerate(sample.get("qa", [])):
        if not isinstance(raw_question, dict):
            continue
        evidence = raw_question.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        turn_index = raw_question.get("turn_index")
        if turn_index is None:
            turn_index = next((evidence_to_index.get(str(item)) for item in evidence if str(item) in evidence_to_index), None)
        answer = raw_question.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(item) for item in answer)
        questions.append(
            {
                "question_id": str(raw_question.get("question_id", question_index)),
                "turn_index": turn_index,
                "question": str(raw_question.get("question", "")),
                "answer": str(answer),
            }
        )
    return {
        "dialogue_id": str(sample.get("sample_id", sample.get("dialogue_id", "dialogue"))),
        "turns": turns,
        "questions": questions,
    }


def load_dialogues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_sample(sample) for sample in rows if isinstance(sample, dict)]


# ---------------------------------------------------------------------------
# Memory bank mechanics
# ---------------------------------------------------------------------------

def lexical_score(query: str, memory: dict[str, Any]) -> float:
    query_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    memory_tokens = set(re.findall(r"[a-zA-Z0-9]+", str(memory.get("text", "")).lower()))
    denominator = math.sqrt(len(query_tokens) * len(memory_tokens))
    return len(query_tokens & memory_tokens) / denominator if denominator else 0.0


def retrieve(query: str, memory_bank: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return sorted(memory_bank, key=lambda memory: lexical_score(query, memory), reverse=True)[:top_k]


def parse_manager_output(text: str) -> list[dict[str, Any]]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Memory Manager output does not contain JSON.")
    payload = json.loads(text[start : end + 1])
    decisions = payload.get("memory", [])
    if not isinstance(decisions, list):
        raise ValueError("Memory Manager output has no memory list.")
    return [item for item in decisions if isinstance(item, dict)]


def apply_decisions(
    memory_bank: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    dialogue_id: str,
    turn_index: int,
    turn: dict[str, Any],
) -> None:
    by_id = {str(memory.get("id", "")): memory for memory in memory_bank}
    next_id = len(memory_bank)
    for decision in decisions:
        event = str(decision.get("event", "NONE")).upper()
        memory_id = str(decision.get("id", ""))
        text = str(decision.get("text", "")).strip()
        if event == "ADD" and text:
            entry = {
                "id": f"{dialogue_id}:m{next_id}",
                "text": text,
                "source_turn": turn_index,
                "timestamp": str(turn.get("timestamp", "")),
            }
            next_id += 1
            memory_bank.append(entry)
            by_id[entry["id"]] = entry
        elif event == "UPDATE" and memory_id in by_id and text:
            by_id[memory_id]["text"] = text
            by_id[memory_id]["source_turn"] = turn_index
        elif event == "DELETE" and memory_id in by_id:
            entry = by_id.pop(memory_id)
            memory_bank.remove(entry)


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

    def generate(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
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
        return self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()

    def log_probability(self, prompt: str, completion: str) -> torch.Tensor:
        """Token-level log probabilities over the completion region, shape [L'].

        Uses the same tokenization (add_special_tokens=False) as generate(),
        so the importance ratios are computed under the actual sampling
        distribution.
        """
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        completion_ids = self.tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if completion_ids.numel() == 0:
            return torch.zeros((0,), device=self.device)
        input_ids = torch.cat([prompt_ids, completion_ids]).unsqueeze(0).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        start = max(prompt_ids.numel() - 1, 0)
        completion_logits = logits[:, start : start + completion_ids.numel(), :]
        completion_labels = labels[:, start : start + completion_ids.numel()]
        return F.log_softmax(completion_logits, dim=-1).gather(2, completion_labels.unsqueeze(-1)).squeeze(-1)


def answer_with_model(answer_model: TextModel, question: str, memories: list[dict[str, Any]], max_new_tokens: int) -> str:
    prompt = build_answer_input(question, {"Memory Bank": memories})
    raw = answer_model.generate(prompt, max_new_tokens, temperature=0.0)
    return raw.rsplit("Answer:", 1)[-1].strip()


# ---------------------------------------------------------------------------
# RL update
# ---------------------------------------------------------------------------

def apply_policy_update(
    manager: TextModel,
    reference: TextModel | None,
    optimizer: torch.optim.Optimizer,
    trajectories: list[dict[str, Any]],
    rewards: list[float],
    args: argparse.Namespace,
) -> float:
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=manager.device)
    advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std(unbiased=False) + 1e-6)
    action_losses = []
    for trajectory, advantage in zip(trajectories, advantages):
        for prompt, completion, old_log_prob in trajectory["actions"]:
            token_log_probs = manager.log_probability(prompt, completion)  # [L']
            if token_log_probs.numel() == 0:
                continue
            current_log_prob = token_log_probs.sum()
            ratio = torch.exp(current_log_prob - old_log_prob)
            if args.algorithm == "ppo":
                clipped = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon)
                action_losses.append(-torch.minimum(ratio * advantage, clipped * advantage))
            else:
                action_losses.append(-advantage * current_log_prob)
            if reference is not None:
                with torch.no_grad():
                    reference_log_probs = reference.log_probability(prompt, completion)
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
    (extract -> retrieve -> manager op -> apply); then answer the linked
    questions with the frozen Answer Agent over the resulting bank. One
    GRPO/PPO update per tuple over its num_generations trajectories.
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
            valid = True
            for local_index, turn in enumerate(turns):
                fact = {
                    "speaker": str(turn.get("speaker", "")),
                    "timestamp": str(turn.get("timestamp", "")),
                    "text": str(turn.get("text", "")).strip(),
                }
                if not fact["text"]:
                    continue
                old_memory = retrieve(fact["text"], memory, args.manager_top_k)
                prompt = build_manager_input(old_memory, [fact])
                completion = manager.generate(prompt, args.max_new_tokens, args.temperature)
                with torch.no_grad():
                    old_log_prob = manager.log_probability(prompt, completion).detach().sum()
                actions.append((prompt, completion, old_log_prob))
                try:
                    decisions = parse_manager_output(completion)
                    apply_decisions(memory, decisions, item["dialogue_id"], int(turn.get("turn_index", local_index)), turn)
                except Exception:
                    valid = False
                    break
            trajectories.append({"memory": memory, "actions": actions, "valid": valid})

        rewards = []
        for trajectory in trajectories:
            if not trajectory["valid"] or not trajectory["actions"]:
                rewards.append(0.0)
                continue
            question_rewards = []
            for question in questions:
                retrieved = retrieve(question["question"], trajectory["memory"], args.answer_top_k)
                prediction = answer_with_model(answer, question["question"], retrieved, args.answer_max_new_tokens)
                question_rewards.append(reward_fn(prediction, question["answer"]))
            rewards.append(sum(question_rewards) / len(question_rewards))

        loss = apply_policy_update(manager, reference, optimizer, trajectories, rewards, args)
        if math.isnan(loss):
            continue
        progress.set_postfix(loss=loss, reward=float(torch.tensor(rewards).mean()))


def train_from_dialogues(args, manager, reference, answer, optimizer, reward_fn, dialogues) -> None:
    """Fallback: raw LoCoMo dialogues as whole-dialogue tuples (Algorithm 5).

    Per dialogue: M <- {}; replay every turn; answer all questions of the
    dialogue with the frozen Answer Agent; one update per dialogue.
    """
    progress = tqdm(dialogues, desc=f"memory-manager-{args.algorithm}")
    for dialogue in progress:
        questions = [item for item in dialogue.get("questions", []) if item.get("question")]
        if not questions:
            continue

        trajectories = [{"memory": [], "actions": [], "valid": True} for _ in range(args.num_generations)]
        for turn_index, turn in enumerate(dialogue.get("turns", [])):
            fact = {
                "speaker": str(turn.get("speaker", "")),
                "timestamp": str(turn.get("timestamp", "")),
                "text": str(turn.get("text", "")).strip(),
            }
            if not fact["text"]:
                continue
            for trajectory in trajectories:
                if not trajectory["valid"]:
                    continue
                old_memory = retrieve(fact["text"], trajectory["memory"], args.manager_top_k)
                prompt = build_manager_input(old_memory, [fact])
                completion = manager.generate(prompt, args.max_new_tokens, args.temperature)
                with torch.no_grad():
                    old_log_prob = manager.log_probability(prompt, completion).detach().sum()
                trajectory["actions"].append((prompt, completion, old_log_prob))
                try:
                    decisions = parse_manager_output(completion)
                    apply_decisions(trajectory["memory"], decisions, str(dialogue["dialogue_id"]), turn_index, turn)
                except Exception:
                    trajectory["valid"] = False

        rewards = []
        for trajectory in trajectories:
            if not trajectory["valid"] or not trajectory["actions"]:
                rewards.append(0.0)
                continue
            question_rewards = []
            for question in questions:
                retrieved = retrieve(question["question"], trajectory["memory"], args.answer_top_k)
                prediction = answer_with_model(answer, question["question"], retrieved, args.answer_max_new_tokens)
                question_rewards.append(reward_fn(prediction, question["answer"]))
            rewards.append(sum(question_rewards) / len(question_rewards))

        loss = apply_policy_update(manager, reference, optimizer, trajectories, rewards, args)
        if math.isnan(loss):
            continue
        progress.set_postfix(loss=loss, reward=float(torch.tensor(rewards).mean()))


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
    if is_tuple_data(rows):
        print("Algorithm 1 tuple data detected; replaying dialogue windows per tuple...")
        tuples = normalize_tuples(rows)
        for epoch in range(args.epochs):
            train_from_tuples(args, manager, reference, answer, optimizer, reward_fn, tuples)
            save_checkpoint(manager, Path(args.output_dir), epoch)
    else:
        print("Raw dialogue data detected; training full-dialogue rollouts...")
        dialogues = load_dialogues(rows)
        for epoch in range(args.epochs):
            train_from_dialogues(args, manager, reference, answer, optimizer, reward_fn, dialogues)
            save_checkpoint(manager, Path(args.output_dir), epoch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithm 5: train Memory-R1 Memory Manager.")
    parser.add_argument("--data-path", required=True, help="Algorithm 1 tuples (build_manager_training_data.py output) or raw LoCoMo JSON")
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
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
