"""Build Answer tuples by replaying the same windows as Manager training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.chat import DEFAULT_MAX_PROMPT_TOKENS
from memfactory.locomo_split import SPLIT_DIALOGUE_IDS
from memfactory.run_metadata import file_sha256
from memfactory.memory_runtime import (
    apply_decisions,
    parse_manager_output,
    retrieve,
    retrieve_per_speaker,
)
from scripts.construct_memory_bank import Manager


def read_rows(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(source.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def raw_questions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a raw QA list when inspecting dataset answers."""
    questions = sample.get("qa", sample.get("questions", []))
    result = []
    for index, question in enumerate(questions):
        if not isinstance(question, dict) or not question.get("question"):
            continue
        answer = question.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(item) for item in answer)
        result.append({
            "question_id": str(question.get("question_id", index)),
            "question": str(question["question"]),
            "answer": str(answer),
        })
    return result


def replay_manager_window(
    tuple_row: dict[str, Any], manager: Manager, manager_top_k: int
) -> tuple[list[dict[str, Any]], int]:
    """Replay one Algorithm-5 window from an empty bank using cached facts."""
    bank: list[dict[str, Any]] = []
    invalid_actions = 0
    for turn in tuple_row["dialogue_turns"]:
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
        visible = retrieve(query, bank, manager_top_k)
        raw = manager.generate(bank, facts, manager_top_k)
        try:
            apply_decisions(
                bank,
                parse_manager_output(raw),
                str(tuple_row["dialogue_id"]),
                turn,
                visible_memory_ids={entry["id"] for entry in visible},
            )
        except (ValueError, KeyError, TypeError):
            # Match the Manager rollout's invalid-action transition: no-op.
            invalid_actions += 1
    return bank, invalid_actions


def build_rows(
    manager_tuples: list[dict[str, Any]],
    manager: Manager,
    manager_top_k: int,
    per_speaker_top_k: int,
    split: str,
    manager_model: str,
) -> list[dict[str, Any]]:
    """Produce one Answer tuple per eligible QA at its evidence target turn."""
    allowed_ids = set(SPLIT_DIALOGUE_IDS[split])
    rows = []
    for tuple_row in manager_tuples:
        metadata = tuple_row.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        dialogue_id = str(tuple_row["dialogue_id"])
        row_split = tuple_row.get("split", metadata.get("split"))
        if dialogue_id not in allowed_ids or row_split != split:
            raise ValueError(f"Unexpected dialogue or split in Manager tuples: {dialogue_id}:{row_split}")
        questions = tuple_row["linked_questions"]
        if not questions:
            continue
        bank, invalid_actions = replay_manager_window(tuple_row, manager, manager_top_k)
        participants = tuple_row["participants"]
        for question in questions:
            memories = retrieve_per_speaker(
                question["question"], bank, participants, per_speaker_top_k
            )
            rows.append({
                "dialogue_id": dialogue_id,
                "question_id": str(question["question_id"]),
                "question": str(question["question"]),
                "retrieved_memories": memories,
                "participants": participants,
                "answer": str(question["answer"]),
                "metadata": {
                    "split": split,
                    "manager_model": manager_model,
                    "manager_top_k": manager_top_k,
                    "per_speaker_top_k": per_speaker_top_k,
                    "target_turn": tuple_row["turn_index"],
                    "window_turns": len(tuple_row["dialogue_turns"]),
                    "invalid_actions": invalid_actions,
                    "memory_size": len(bank),
                    "source": "manager_tuple_window_replay_v1",
                },
            })
    return rows


def build(
    manager_tuples_path: str,
    output_path: str,
    manager_model: str,
    manager_adapter: str | None,
    manager_device: str,
    manager_top_k: int,
    per_speaker_top_k: int,
    max_new_tokens: int,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
    split: str = "train",
) -> None:
    manager_tuples = read_rows(manager_tuples_path)
    manager = Manager(manager_model, manager_device, max_new_tokens, manager_adapter, max_prompt_tokens)
    rows = build_rows(
        manager_tuples, manager, manager_top_k, per_speaker_top_k, split, manager_model
    )
    if not rows:
        raise ValueError(f"No linked QA in {manager_tuples_path} for split {split}.")
    source_hash = file_sha256(manager_tuples_path)
    for row in rows:
        row["metadata"]["manager_tuples_sha256"] = source_hash
        row["metadata"]["manager_adapter"] = manager_adapter
    write_jsonl(output_path, rows)
    per_window = {
        (row["dialogue_id"], row["metadata"]["target_turn"]): row["metadata"]["invalid_actions"]
        for row in rows
    }
    invalid = sum(per_window.values())
    print(f"wrote {len(rows)} Answer tuples to {output_path}; invalid replay actions={invalid}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Answer tuples from Manager training windows.")
    parser.add_argument("--manager-tuples", required=True, help="Algorithm-1 tuples with cached turn facts.")
    parser.add_argument("--split", required=True, choices=list(SPLIT_DIALOGUE_IDS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--manager-adapter")
    parser.add_argument("--manager-device", default="auto")
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--per-speaker-top-k", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    args = parser.parse_args()
    build(
        args.manager_tuples,
        args.output,
        args.manager_model,
        args.manager_adapter,
        args.manager_device,
        args.manager_top_k,
        args.per_speaker_top_k,
        args.max_new_tokens,
        args.max_prompt_tokens,
        args.split,
    )


if __name__ == "__main__":
    main()
