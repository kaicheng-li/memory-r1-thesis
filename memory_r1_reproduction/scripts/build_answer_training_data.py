"""Algorithm 2: build Answer Agent data from a trained Memory Manager."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
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
from memfactory.locomo_split import select_locomo_split

LOGGER = logging.getLogger("memory_r1.answer_data")


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
    manager_adapter: str | None,
    device: str,
    manager_top_k: int,
    answer_top_k_per_speaker: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    split: str,
) -> None:
    raw_samples = read_json(input_path)
    raw_samples = select_locomo_split(raw_samples, split)
    dialogues = load_dialogues(raw_samples)
    total_turns = sum(len(dialogue["turns"]) for dialogue in dialogues)
    LOGGER.info("loaded dialogues=%d turns=%d", len(dialogues), total_turns)
    manager = Manager(manager_model, device, max_new_tokens, max_prompt_tokens, manager_adapter)
    rows = []
    completed_turns = 0

    for dialogue_index, (raw_sample, dialogue) in enumerate(zip(raw_samples, dialogues), start=1):
        dialogue_id = str(dialogue["dialogue_id"])
        memory_bank: list[dict[str, Any]] = []
        LOGGER.info(
            "dialogue start %d/%d id=%s turns=%d",
            dialogue_index,
            len(dialogues),
            dialogue_id,
            len(dialogue["turns"]),
        )

        for turn_index, turn in enumerate(dialogue["turns"]):
            started = time.monotonic()
            LOGGER.info(
                "turn start dialogue=%s turn=%d/%d overall=%d/%d",
                dialogue_id,
                turn_index + 1,
                len(dialogue["turns"]),
                completed_turns + 1,
                total_turns,
            )
            facts = manager.extract(turn)
            if not facts:
                completed_turns += 1
                LOGGER.info(
                    "turn done dialogue=%s turn=%d/%d facts=0 memory=%d elapsed=%.1fs progress=%d/%d",
                    dialogue_id,
                    turn_index + 1,
                    len(dialogue["turns"]),
                    len(memory_bank),
                    time.monotonic() - started,
                    completed_turns,
                    total_turns,
                )
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
            completed_turns += 1
            LOGGER.info(
                "turn done dialogue=%s turn=%d/%d facts=%d memory=%d elapsed=%.1fs progress=%d/%d",
                dialogue_id,
                turn_index + 1,
                len(dialogue["turns"]),
                len(facts),
                len(memory_bank),
                time.monotonic() - started,
                completed_turns,
                total_turns,
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
                    "category": int(question.get("category", 0)),
                    "metadata": {
                        "manager_model": manager_model,
                        "manager_top_k": manager_top_k,
                        "answer_top_k_per_speaker": answer_top_k_per_speaker,
                        "memory_size": len(memory_bank),
                    },
                }
            )

        LOGGER.info("dialogue done %d/%d id=%s answer_rows=%d", dialogue_index, len(dialogues), dialogue_id, len(rows))

    write_jsonl(output_path, rows)
    LOGGER.info("wrote %d Answer tuples to %s", len(rows), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Algorithm 2 Answer Agent tuples.")
    parser.add_argument("--input", required=True, help="Raw LoCoMo JSON file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manager-model", required=True, help="Trained Memory Manager checkpoint.")
    parser.add_argument("--manager-adapter")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k-per-speaker", type=int, default=30)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    build(
        args.input,
        args.output,
        args.manager_model,
        args.manager_adapter,
        args.device,
        args.manager_top_k,
        args.answer_top_k_per_speaker,
        args.max_prompt_tokens,
        args.max_new_tokens,
        args.split,
    )


if __name__ == "__main__":
    main()
