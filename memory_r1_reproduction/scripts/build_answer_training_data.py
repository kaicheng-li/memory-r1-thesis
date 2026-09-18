"""Algorithm 2: build Answer Agent data from a trained Memory Manager."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.construct_memory_bank import (
    Manager,
    apply_decisions,
    load_dialogues,
    parse_manager_output,
    top_k_per_speaker,
)


def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(
    input_path: str,
    output_path: str,
    manager_model: str,
    device: str,
    manager_top_k: int,
    answer_top_k_per_speaker: int,
    max_new_tokens: int,
) -> None:
    raw_samples = read_json(input_path)
    raw_samples = raw_samples if isinstance(raw_samples, list) else [raw_samples]
    dialogues = load_dialogues(raw_samples)
    manager = Manager(manager_model, device, max_new_tokens)
    rows = []

    for raw_sample, dialogue in zip(raw_samples, dialogues):
        dialogue_id = str(dialogue["dialogue_id"])
        memory_bank: list[dict[str, Any]] = []

        for turn_index, turn in enumerate(dialogue["turns"]):
            facts = manager.extract(turn)
            if not facts:
                continue
            manager_output = manager.generate(memory_bank, facts, manager_top_k)
            apply_decisions(
                memory_bank,
                parse_manager_output(manager_output),
                dialogue_id,
                turn_index=turn_index,
                default_speaker=turn["speaker"],
                timestamp=turn["timestamp"],
            )

        questions = raw_sample.get("qa", raw_sample.get("questions", []))
        for question_index, question in enumerate(questions):
            if not question.get("question"):
                continue
            retrieved_memories = top_k_per_speaker(
                str(question["question"]),
                memory_bank,
                dialogue["participants"],
                answer_top_k_per_speaker,
            )
            rows.append(
                {
                    "dialogue_id": dialogue_id,
                    "participants": dialogue["participants"],
                    "question_id": str(question.get("question_id", question_index)),
                    "question": str(question["question"]),
                    "retrieved_memories": retrieved_memories,
                    "answer": str(question.get("answer", "")),
                    "metadata": {
                        "manager_model": manager_model,
                        "manager_top_k": manager_top_k,
                        "answer_top_k_per_speaker": answer_top_k_per_speaker,
                        "memory_size": len(memory_bank),
                    },
                }
            )

    write_jsonl(output_path, rows)
    print(f"wrote {len(rows)} Answer tuples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Algorithm 2 Answer Agent tuples.")
    parser.add_argument("--input", required=True, help="Raw LoCoMo JSON file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manager-model", required=True, help="Trained Memory Manager checkpoint.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k-per-speaker", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    build(
        args.input,
        args.output,
        args.manager_model,
        args.device,
        args.manager_top_k,
        args.answer_top_k_per_speaker,
        args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
