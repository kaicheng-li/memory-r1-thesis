"""Build Answer Agent tuples from a Memory Manager-produced memory bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.memory_runtime import apply_decisions, parse_manager_output, retrieve_per_speaker
from scripts.construct_memory_bank import Manager, load_dialogues

TOP_K = 60


def read_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def raw_questions(sample: dict[str, Any]) -> list[dict[str, Any]]:
    questions = sample.get("qa", sample.get("questions", []))
    return [
        {
            "question_id": str(question.get("question_id", index)),
            "question": str(question["question"]),
            "answer": str(question.get("answer", "")),
        }
        for index, question in enumerate(questions)
        if question.get("question")
    ]


def build(
    input_path: str,
    output_path: str,
    manager_model: str,
    manager_device: str,
    manager_top_k: int,
    per_speaker_top_k: int,
    max_new_tokens: int,
) -> None:
    raw_samples = read_json(input_path)
    raw_samples = raw_samples if isinstance(raw_samples, list) else [raw_samples]
    dialogues = load_dialogues(raw_samples)
    manager = Manager(manager_model, manager_device, max_new_tokens)
    rows = []

    for raw_sample, dialogue in zip(raw_samples, dialogues):
        dialogue_id = str(dialogue["dialogue_id"])
        memory_bank: list[dict[str, Any]] = []

        for turn in dialogue["turns"]:
            facts = manager.extract(turn)
            if not facts:
                continue
            manager_output = manager.generate(memory_bank, facts, manager_top_k)
            decisions = parse_manager_output(manager_output)
            apply_decisions(memory_bank, decisions, dialogue_id, turn)

        for question in raw_questions(raw_sample):
            memories = retrieve_per_speaker(
                question["question"],
                memory_bank,
                dialogue["participants"],
                per_speaker_top_k,
            )
            rows.append(
                {
                    "dialogue_id": dialogue_id,
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "retrieved_memories": memories,
                    "answer": question["answer"],
                    "metadata": {
                        "manager_model": manager_model,
                        "manager_top_k": manager_top_k,
                        "per_speaker_top_k": per_speaker_top_k,
                        "memory_size": len(memory_bank),
                    },
                }
            )

    write_jsonl(output_path, rows)
    print(f"wrote {len(rows)} Answer tuples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Answer Agent tuples from Manager-generated memories.")
    parser.add_argument("--input", required=True, help="Raw LoCoMo JSON file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manager-model", required=True, help="Memory Manager checkpoint.")
    parser.add_argument("--manager-device", default="auto")
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--per-speaker-top-k", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    build(
        args.input,
        args.output,
        args.manager_model,
        args.manager_device,
        args.manager_top_k,
        args.per_speaker_top_k,
        args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
