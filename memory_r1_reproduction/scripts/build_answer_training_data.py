"""Build Answer Agent training tuples (paper: Answer Agent Training Data).

For every question linked to a turn in the Algorithm 1 tuples, retrieve the
60 most relevant memories from that turn's teacher-built temporal memory bank
(GPT-4o-mini), and emit (question, retrieved_memories, gold answer) tuples.

No Memory Manager is replayed here: the temporal banks already exist as the
Algorithm 1 output.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

TOP_K = 60


def read_rows(path: str) -> list[dict[str, Any]]:
    """Read a JSONL or JSON dataset into a list of rows."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_score(question: str, memory: dict[str, Any]) -> float:
    question_tokens = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
    memory_tokens = set(re.findall(r"[a-zA-Z0-9]+", memory.get("text", "").lower()))
    denominator = math.sqrt(len(question_tokens) * len(memory_tokens))
    return len(question_tokens & memory_tokens) / denominator if denominator else 0.0


def retrieve_memories(question: str, memory_bank: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    return sorted(memory_bank, key=lambda memory: word_score(question, memory), reverse=True)[:top_k]


def build(input_path: str, output_path: str, retrieval_top_k: int) -> None:
    rows = []
    for row in read_rows(input_path):
        if not isinstance(row, dict):
            continue
        bank = row.get("temporal_memory_bank", [])
        if not isinstance(bank, list):
            continue
        for question in row.get("linked_questions", []):
            if not isinstance(question, dict) or not question.get("question"):
                continue
            candidates = retrieve_memories(question["question"], bank, retrieval_top_k)
            rows.append(
                {
                    "dialogue_id": str(row.get("dialogue_id", "dialogue")),
                    "question_id": str(question.get("question_id", "")),
                    "question": question["question"],
                    "retrieved_memories": candidates,
                    "answer": str(question.get("answer", "")),
                    "metadata": {
                        "bank_turn": int(row.get("turn_index", 0)),
                        "top_k": retrieval_top_k,
                        "retrieved_count": len(candidates),
                    },
                }
            )
    write_jsonl(output_path, rows)
    print(f"wrote {len(rows)} Answer tuples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Answer Agent training tuples from Algorithm 1 output.")
    parser.add_argument("--input", required=True, help="Algorithm 1 tuples (build_manager_training_data.py output)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--retrieval-top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    build(args.input, args.output, args.retrieval_top_k)


if __name__ == "__main__":
    main()
