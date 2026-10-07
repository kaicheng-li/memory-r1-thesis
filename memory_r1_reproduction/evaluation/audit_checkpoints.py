"""Offline, checkpoint-based audit for Memory-R1 Manager and Answer adapters.

This script does not rely on training logs. It evaluates one checkpoint per
process and writes both per-example JSONL and an aggregate JSON summary, so
base/adapters can be compared on exactly the same deterministically selected
examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.chat import tokenize_chat_prompt
from evaluation.metrics import (
    add_question_metadata,
    aggregate_answer_metrics,
    bleu1_score,
    exact_match,
    f1_score,
    load_question_metadata,
)
from memfactory.locomo_split import question_keys
from memfactory.lora import load_lora_model
from memfactory.modules.memory_retriever import build_answer_input
from memfactory.modules.memory_updater import build_manager_input


def read_rows(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(source.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def extract_answer(text: str) -> str:
    tagged = re.search(r"<answer>(.*?)</answer>", text, re.IGNORECASE | re.DOTALL)
    if tagged:
        return tagged.group(1).strip()
    marked = re.search(r"\*\*Answer:\*\*\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if marked:
        return marked.group(1).strip()
    answer = re.search(r"Answer:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    return answer.group(1).strip() if answer else text.strip()


def manager_instances_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    instances = []
    for row in rows:
        current_turn = row.get("current_turn")
        if not isinstance(current_turn, dict):
            continue
        dialogue_turns = [
            turn
            for turn in row.get("dialogue_turns", [])
            if isinstance(turn, dict) and str(turn.get("text", "")).strip()
        ]
        base = {
            "dialogue_id": str(row.get("dialogue_id", "dialogue")),
            "participants": row.get("participants", []),
            "turn_index": int(row.get("turn_index", 0)),
            "dialogue_turns": dialogue_turns,
            "current_turn": current_turn,
        }
        for question in row.get("linked_questions", []):
            if isinstance(question, dict) and question.get("question"):
                instances.append({**base, "question": question})
    return instances


def f1_reward(prediction: str, gold: str) -> float:
    return f1_score(prediction, gold)


def stable_seed(base_seed: int, *parts: object) -> int:
    material = "|".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big")


def stable_sample(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count <= 0 or count >= len(rows):
        return rows

    def key(row: dict[str, Any]) -> bytes:
        question = row.get("question", "")
        if isinstance(question, dict):
            question_id = question.get("question_id", "")
            question_text = question.get("question", "")
        else:
            question_id = row.get("question_id", "")
            question_text = question
        identity = (
            f"{seed}|{row.get('dialogue_id', '')}|{row.get('turn_index', '')}|"
            f"{question_id}|{question_text}"
        )
        return hashlib.sha256(identity.encode("utf-8")).digest()

    return sorted(rows, key=key)[:count]


def latency_summary(
    records: list[dict[str, Any]], components: dict[str, str]
) -> dict[str, dict[str, float | int | None]]:
    """Summarize per-call wall time using the paper's p50/p95 convention."""

    result = {}
    for component, key in components.items():
        values = []
        for record in records:
            value = record.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
            for action in record.get("actions", []):
                value = action.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
        values.sort()

        def percentile(fraction: float) -> float | None:
            if not values:
                return None
            index = max(0, min(len(values) - 1, int(fraction * len(values) + 0.999999) - 1))
            return values[index]

        result[component] = {
            "count": len(values),
            "mean_s": sum(values) / len(values) if values else None,
            "p50_s": percentile(0.50),
            "p95_s": percentile(0.95),
        }
    return result


class AuditModel:
    def __init__(self, model_path: str, adapter_path: str | None, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        self.model = load_lora_model(
            model_path, device, adapter_path=adapter_path, trainable=False
        )
        self.model.eval()
        self.last_elapsed_s = 0.0

    def generate(
        self,
        prompt: str,
        max_prompt_tokens: int,
        max_new_tokens: int,
        temperature: float,
        seed: int,
    ) -> tuple[str, int, bool]:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        inputs = tokenize_chat_prompt(
            self.tokenizer, prompt, max_prompt_tokens
        ).to(self.device)
        generation_options = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if temperature > 0:
            generation_options["temperature"] = temperature
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_options)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        self.last_elapsed_s = time.perf_counter() - started
        prompt_length = inputs["input_ids"].shape[1]
        completion_ids = output[0][prompt_length:]
        token_count = int(completion_ids.numel())
        hit_token_limit = token_count >= max_new_tokens
        text = self.tokenizer.decode(
            completion_ids, skip_special_tokens=True
        ).strip()
        return text, token_count, hit_token_limit


def parse_manager_payload(text: str) -> list[dict[str, Any]]:
    # Keep offline audit parsing identical to Algorithm 3/4 construction.
    # Both paths accept a JSON object containing a list under ``memory`` and
    # tolerate assistant prefixes/fences before the object.
    from scripts.construct_memory_bank import parse_manager_output

    return parse_manager_output(text)


def operations_are_valid(
    decisions: list[dict[str, Any]], memory: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    existing_ids = {str(item.get("id", "")) for item in memory}
    errors = []
    for index, decision in enumerate(decisions):
        event = str(decision.get("event", "NONE")).upper()
        memory_id = str(decision.get("id", ""))
        text = str(decision.get("text", "")).strip()
        if event not in {"ADD", "UPDATE", "DELETE", "NONE"}:
            errors.append(f"item {index}: unknown event {event!r}")
        elif event == "ADD" and not text:
            errors.append(f"item {index}: ADD has no text")
        elif event == "UPDATE" and (memory_id not in existing_ids or not text):
            errors.append(f"item {index}: UPDATE id/text is not applicable")
        elif event == "DELETE" and memory_id not in existing_ids:
            errors.append(f"item {index}: DELETE id is not applicable")
    return not errors, errors


def answer_prompt(row: dict[str, Any]) -> str:
    memories = row.get("retrieved_memories", row.get("memories", []))
    participants = [str(value) for value in row.get("participants", [])]
    memories_by_speaker = {
        participant: [
            memory
            for memory in memories
            if str(memory.get("speaker", "")) == participant
        ]
        for participant in participants
    }
    return build_answer_input(str(row["question"]), memories_by_speaker)


def audit_answer(args: argparse.Namespace) -> dict[str, Any]:
    question_metadata = load_question_metadata(args.raw_data_path)
    allowed = question_keys(args.raw_data_path, args.split) if args.raw_data_path else None
    rows = []
    for row in read_rows(args.data_path):
        add_question_metadata(row, question_metadata)
        if allowed is not None and (
            str(row.get("dialogue_id", "")), str(row.get("question_id", ""))
        ) not in allowed:
            continue
        memories = row.get("retrieved_memories", row.get("memories", []))
        gold = str(row.get("answer", row.get("gold", ""))).strip()
        question = str(row.get("question", "")).strip()
        if question and gold and isinstance(memories, list) and memories:
            rows.append(row)
    selected = stable_sample(rows, args.sample_count, args.seed)
    if not selected:
        raise ValueError(
            f"Answer audit has no usable rows after split={args.split!r}. "
            f"Check ANSWER_DATA_PATH was built for this dialogue split and contains retrieved memories."
        )
    model = AuditModel(args.model_path, args.adapter, args.device)
    records = []
    for row_index, row in enumerate(selected):
        print(
            f"answer audit {args.label}: QA {row_index + 1}/{len(selected)}",
            flush=True,
        )
        prompt = answer_prompt(row)
        gold = str(row.get("answer", row.get("gold", "")))
        for generation_index in range(args.num_generations):
            raw, token_count, hit_limit = model.generate(
                prompt,
                args.max_prompt_tokens,
                args.max_new_tokens,
                args.temperature,
                stable_seed(args.seed, "answer", row_index, generation_index),
            )
            prediction = extract_answer(raw)
            records.append(
                {
                    "component": "answer",
                    "label": args.label,
                    "sample_index": row_index,
                    "dialogue_id": row.get("dialogue_id"),
                    "question_id": row.get("question_id"),
                    "question": row.get("question"),
                    "gold": gold,
                    "category": row.get("category", 0),
                    "category_name": row.get("category_name", "unknown"),
                    "generation": generation_index,
                    "prediction": prediction,
                    "raw_output": raw,
                    "completion_tokens": token_count,
                    "answer_elapsed_s": model.last_elapsed_s,
                    "hit_token_limit": hit_limit,
                    "em": exact_match(prediction, gold),
                    "f1": f1_reward(prediction, gold),
                    "bleu1": bleu1_score(prediction, gold),
                }
            )

    count = len(records)
    answer_groups = [
        [record for record in records if record["sample_index"] == row_index]
        for row_index in range(len(selected))
    ]
    return {
        "component": "answer",
        "label": args.label,
        "metric_scorer": "memory_r1_locomo_v1",
        "adapter": args.adapter,
        "data_path": args.data_path,
        "split": args.split,
        "available_rows_with_memory": len(rows),
        "selected_rows": len(selected),
        "generations": count,
        "temperature": args.temperature,
        "em": sum(row["em"] for row in records) / count if count else 0.0,
        "f1": sum(row["f1"] for row in records) / count if count else 0.0,
        "bleu1": sum(row["bleu1"] for row in records) / count if count else 0.0,
        "paper_metrics": aggregate_answer_metrics(records, ("f1", "bleu1")),
        "latency": latency_summary(records, {"answer_agent": "answer_elapsed_s"}),
        "nonzero_rewards": sum(row["em"] > 0 for row in records),
        "reward_signal_groups": sum(
            len({record["em"] for record in group}) > 1 for group in answer_groups
        ),
        "reward_groups": len(answer_groups),
        "nonempty_outputs": sum(bool(row["prediction"]) for row in records),
        "hit_token_limit": sum(row["hit_token_limit"] for row in records),
        "records": records,
    }


def manager_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_rows(args.data_path)
    allowed = question_keys(args.raw_data_path, args.split)
    question_metadata = load_question_metadata(args.raw_data_path)
    for row in rows:
        dialogue_id = str(row.get("dialogue_id", "dialogue"))
        linked_questions = []
        for question in row.get("linked_questions", []):
            question = dict(question)
            question["dialogue_id"] = dialogue_id
            add_question_metadata(question, question_metadata)
            if (dialogue_id, str(question.get("question_id", ""))) in allowed:
                linked_questions.append(question)
        row["linked_questions"] = linked_questions
    return manager_instances_from_rows(rows)


def audit_manager(args: argparse.Namespace) -> dict[str, Any]:
    # The embedding retriever is only required for Manager replay. Keeping the
    # import here avoids loading sentence-transformers for --help/Answer audit.
    from scripts.construct_memory_bank import (
        COMPACT_MANAGER_RETRY,
        apply_decisions,
        top_k,
        top_k_per_speaker,
    )

    instances = manager_instances(args)
    selected = stable_sample(instances, args.sample_count, args.seed)
    if not selected:
        raise ValueError(
            f"Manager audit has no linked QA after split={args.split!r}. "
            "Check MANAGER_DATA_PATH was built for this dialogue split."
        )
    manager = AuditModel(args.model_path, args.adapter, args.device)
    answer = AuditModel(args.model_path, args.answer_adapter, args.device)
    records = []

    for row_index, item in enumerate(selected):
        print(
            f"manager audit {args.label}: QA {row_index + 1}/{len(selected)}",
            flush=True,
        )
        question = item["question"]
        for generation_index in range(args.num_generations):
            memory: list[dict[str, Any]] = []
            actions = []
            trajectory_complete = True
            expected_actions = sum(
                bool(turn.get("facts")) for turn in item["dialogue_turns"]
            )
            for turn_index, turn in enumerate(item["dialogue_turns"]):
                facts = [
                    {
                        "speaker": str(fact.get("speaker", turn.get("speaker", ""))),
                        "timestamp": str(fact.get("timestamp", turn.get("timestamp", ""))),
                        "key": str(fact.get("key", "")),
                        "memory_type": str(fact.get("memory_type", "")),
                        "tags": fact.get("tags", []),
                        "text": str(fact.get("text", "")).strip(),
                    }
                    for fact in turn.get("facts", [])
                    if isinstance(fact, dict) and str(fact.get("text", "")).strip()
                ]
                if not facts:
                    continue
                query = " ".join(fact["text"] for fact in facts)
                search_started = time.perf_counter()
                old_memory = top_k(query, memory, args.manager_top_k)
                search_elapsed = time.perf_counter() - search_started
                prompt = build_manager_input(old_memory, facts)
                raw, token_count, hit_limit = manager.generate(
                    prompt,
                    args.max_prompt_tokens,
                    args.max_new_tokens,
                    args.temperature,
                    stable_seed(
                        args.seed,
                        "manager",
                        row_index,
                        generation_index,
                        turn_index,
                    ),
                )
                action = {
                    "turn_index": int(turn.get("turn_index", turn_index)),
                    "completion_tokens": token_count,
                    "hit_token_limit": hit_limit,
                    "manager_attempts": 1,
                    "manager_elapsed_s": manager.last_elapsed_s,
                    "search_elapsed_s": search_elapsed,
                    "json_valid": False,
                    "operations_valid": False,
                }
                try:
                    decisions = parse_manager_payload(raw)
                    action["json_valid"] = True
                    semantic_valid, semantic_errors = operations_are_valid(
                        decisions, old_memory
                    )
                    action["operations_valid"] = semantic_valid
                    action["operation_errors"] = semantic_errors
                    apply_decisions(
                        memory,
                        decisions,
                        str(item["dialogue_id"]),
                        int(turn.get("turn_index", turn_index)),
                        str(turn.get("speaker", "")),
                        str(turn.get("timestamp", "")),
                    )
                except (ValueError, json.JSONDecodeError) as exc:
                    # Match construct_memory_bank.py: a truncated/non-JSON
                    # response gets a compact, operation-only retry before the
                    # trajectory is declared invalid.
                    action["initial_parse_error"] = str(exc)
                    action["initial_raw_output"] = raw
                    retry_prompt = prompt + "\n\n" + COMPACT_MANAGER_RETRY
                    retry_raw, retry_tokens, retry_hit_limit = manager.generate(
                        retry_prompt,
                        args.max_prompt_tokens,
                        args.max_new_tokens,
                        args.temperature,
                        stable_seed(
                            args.seed,
                            "manager-retry",
                            row_index,
                            generation_index,
                            turn_index,
                        ),
                    )
                    action["manager_attempts"] = 2
                    action["completion_tokens"] = retry_tokens
                    action["hit_token_limit"] = hit_limit or retry_hit_limit
                    action["manager_elapsed_s"] += manager.last_elapsed_s
                    try:
                        decisions = parse_manager_payload(retry_raw)
                        action["json_valid"] = True
                        semantic_valid, semantic_errors = operations_are_valid(
                            decisions, old_memory
                        )
                        action["operations_valid"] = semantic_valid
                        action["operation_errors"] = semantic_errors
                        apply_decisions(
                            memory,
                            decisions,
                            str(item["dialogue_id"]),
                            int(turn.get("turn_index", turn_index)),
                            str(turn.get("speaker", "")),
                            str(turn.get("timestamp", "")),
                        )
                    except (ValueError, json.JSONDecodeError) as retry_exc:
                        trajectory_complete = False
                        action["parse_error"] = str(retry_exc)
                        action["raw_output"] = retry_raw
                        actions.append(action)
                        break
                actions.append(action)

            prediction = ""
            em = 0.0
            f1 = 0.0
            bleu1 = 0.0
            answer_elapsed = None
            final_search_elapsed = None
            if trajectory_complete and actions:
                search_started = time.perf_counter()
                retrieved = top_k_per_speaker(
                    str(question["question"]),
                    memory,
                    item["participants"],
                    args.answer_top_k_per_speaker,
                )
                final_search_elapsed = time.perf_counter() - search_started
                prompt = build_answer_input(
                    str(question["question"]),
                    {
                        participant: [
                            entry
                            for entry in retrieved
                            if entry["speaker"] == participant
                        ]
                        for participant in item["participants"]
                    },
                )
                raw_answer, _, _ = answer.generate(
                    prompt,
                    args.max_prompt_tokens,
                    args.answer_max_new_tokens,
                    args.answer_temperature,
                    stable_seed(args.seed, "manager-answer", row_index, generation_index),
                )
                answer_elapsed = answer.last_elapsed_s
                prediction = extract_answer(raw_answer)
                em = exact_match(prediction, str(question["answer"]))
                f1 = f1_reward(prediction, str(question["answer"]))
                bleu1 = bleu1_score(prediction, str(question["answer"]))

            records.append(
                {
                    "component": "manager",
                    "label": args.label,
                    "sample_index": row_index,
                    "dialogue_id": item["dialogue_id"],
                    "question_id": question.get("question_id"),
                    "question": question["question"],
                    "gold": question["answer"],
                    "category": question.get("category", 0),
                    "category_name": question.get("category_name", "unknown"),
                    "generation": generation_index,
                    "expected_actions": expected_actions,
                    "attempted_actions": len(actions),
                    "json_valid_actions": sum(action["json_valid"] for action in actions),
                    "operations_valid_actions": sum(
                        action["operations_valid"] for action in actions
                    ),
                    "hit_token_limit_actions": sum(
                        action["hit_token_limit"] for action in actions
                    ),
                    "trajectory_complete": trajectory_complete,
                    "final_memory_size": len(memory),
                    "prediction": prediction,
                    "em": em,
                    "f1": f1,
                    "bleu1": bleu1,
                    "answer_elapsed_s": answer_elapsed,
                    "final_search_elapsed_s": final_search_elapsed,
                    "actions": actions,
                }
            )

    trajectories = len(records)
    attempted_actions = sum(row["attempted_actions"] for row in records)
    complete_records = [row for row in records if row["trajectory_complete"]]
    complete_count = len(complete_records)
    manager_groups = [
        [record for record in records if record["sample_index"] == row_index]
        for row_index in range(len(selected))
    ]
    return {
        "component": "manager",
        "label": args.label,
        "metric_scorer": "memory_r1_locomo_v1",
        "adapter": args.adapter,
        "answer_adapter": args.answer_adapter,
        "data_path": args.data_path,
        "split": args.split,
        "available_QA": len(instances),
        "selected_QA": len(selected),
        "trajectories": trajectories,
        "temperature": args.temperature,
        "expected_actions": sum(row["expected_actions"] for row in records),
        "attempted_actions": attempted_actions,
        "json_valid_actions": sum(row["json_valid_actions"] for row in records),
        "json_valid_rate": (
            sum(row["json_valid_actions"] for row in records) / attempted_actions
            if attempted_actions
            else 0.0
        ),
        "operations_valid_actions": sum(
            row["operations_valid_actions"] for row in records
        ),
        "hit_token_limit_actions": sum(
            row["hit_token_limit_actions"] for row in records
        ),
        "complete_trajectories": complete_count,
        "complete_trajectory_rate": complete_count / trajectories if trajectories else 0.0,
        "em_all": sum(row["em"] for row in records) / trajectories if trajectories else 0.0,
        "f1_all": sum(row["f1"] for row in records) / trajectories if trajectories else 0.0,
        "bleu1_all": (
            sum(row["bleu1"] for row in records) / trajectories if trajectories else 0.0
        ),
        "paper_metrics_all": aggregate_answer_metrics(records, ("f1", "bleu1")),
        "nonzero_rewards": sum(row["em"] > 0 for row in records),
        "reward_signal_groups": sum(
            len({record["em"] for record in group}) > 1 for group in manager_groups
        ),
        "reward_groups": len(manager_groups),
        "em_complete": (
            sum(row["em"] for row in complete_records) / complete_count
            if complete_count
            else 0.0
        ),
        "f1_complete": (
            sum(row["f1"] for row in complete_records) / complete_count
            if complete_count
            else 0.0
        ),
        "bleu1_complete": (
            sum(row["bleu1"] for row in complete_records) / complete_count
            if complete_count
            else 0.0
        ),
        "paper_metrics_complete": aggregate_answer_metrics(
            complete_records, ("f1", "bleu1")
        ),
        "latency": latency_summary(
            records,
            {
                "memory_manager": "manager_elapsed_s",
                "memory_search": "search_elapsed_s",
                "answer_agent": "answer_elapsed_s",
                "final_memory_search": "final_search_elapsed_s",
            },
        ),
        "records": records,
    }


def write_report(report: dict[str, Any], output_path: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = report.pop("records")
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    records_path = output.with_suffix(".jsonl")
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote summary to {output}", flush=True)
    print(f"wrote records to {records_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=["answer", "manager"])
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", help="Adapter under audit; omit for the base model")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--raw-data-path",
        help="Raw LoCoMo JSON used for split filtering and authoritative QA categories",
    )
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--answer-adapter")
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--answer-max-new-tokens", type=int, default=512)
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k-per-speaker", type=int, default=30)
    args = parser.parse_args()
    if args.sample_count < 0 or args.num_generations < 1:
        parser.error("sample-count must be non-negative and num-generations at least 1")
    if args.component == "manager" and not args.raw_data_path:
        parser.error("manager audit requires --raw-data-path")

    report = audit_answer(args) if args.component == "answer" else audit_manager(args)
    write_report(report, args.output)


if __name__ == "__main__":
    main()
