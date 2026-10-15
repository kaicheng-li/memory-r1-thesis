"""Build Answer-Agent tuples by rolling the selected Manager over cached facts."""

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

from evaluation.manager_validation import manager_transition
from scripts.construct_memory_bank import Manager, top_k_per_speaker

LOGGER = logging.getLogger("memory_r1.answer_data")


def read_timelines(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    timelines = payload if isinstance(payload, list) else [payload]
    if not timelines:
        raise ValueError(f"No timelines found in {path}")
    return timelines


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args: argparse.Namespace) -> None:
    timelines = read_timelines(args.timeline)
    total_turns = sum(len(item["turns"]) for item in timelines)
    manager = Manager(
        args.manager_model,
        args.device,
        args.max_new_tokens,
        args.max_prompt_tokens,
        args.manager_adapter,
    )
    rows = []
    completed_turns = 0
    trace_path = Path(args.output).with_suffix(".manager_trace.jsonl")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("", encoding="utf-8")
    LOGGER.info("Manager turn trace: %s", trace_path)
    for dialogue_index, timeline in enumerate(timelines, start=1):
        dialogue_id = str(timeline["dialogue_id"])
        participants = [str(item) for item in timeline["participants"]]
        memory_bank: list[dict[str, Any]] = []
        invalid_turns = 0
        for turn in timeline["turns"]:
            started = time.monotonic()
            turn_index = int(turn["turn_index"])
            facts = [
                fact for fact in turn.get("facts", [])
                if isinstance(fact, dict) and str(fact.get("text", "")).strip()
            ]
            LOGGER.info(
                "turn start dialogue=%s turn=%d/%d facts=%d memory=%d",
                dialogue_id, turn_index + 1, len(timeline["turns"]),
                len(facts), len(memory_bank),
            )
            memory_bank, action = manager_transition(
                {**turn, "dialogue_id": dialogue_id, "current_facts": facts},
                memory_bank,
                manager.generate_prompt,
                args.manager_top_k,
            )
            if not action["valid"]:
                invalid_turns += 1
                LOGGER.warning(
                    "Manager action rejected; bank unchanged dialogue=%s "
                    "turn=%d error=%s raw=%r",
                    dialogue_id, turn_index + 1, action["error"], action.get("raw"),
                )
            with trace_path.open("a", encoding="utf-8") as trace:
                trace.write(json.dumps({
                    "dialogue_id": dialogue_id,
                    "turn_index": turn_index,
                    "memory_size": len(memory_bank),
                    "elapsed_s": time.monotonic() - started,
                    **action,
                }, ensure_ascii=False) + "\n")
            completed_turns += 1
            LOGGER.info(
                "turn done dialogue=%s turn=%d/%d facts=%d memory=%d "
                "elapsed=%.1fs progress=%d/%d valid=%s actions=%d invalid_turns=%d",
                dialogue_id,
                turn_index + 1,
                len(timeline["turns"]),
                len(facts),
                len(memory_bank),
                time.monotonic() - started,
                completed_turns,
                total_turns,
                action["valid"],
                action["actions"],
                invalid_turns,
            )

        for question in timeline["questions"]:
            retrieved = top_k_per_speaker(
                question["question"], memory_bank, participants,
                args.answer_top_k_per_speaker,
            )
            rows.append({
                "dialogue_id": dialogue_id,
                "participants": participants,
                "question_id": str(question["question_id"]),
                "question": str(question["question"]),
                "retrieved_memories": retrieved,
                "answer": str(question["answer"]),
                "category": int(question["category"]),
                "metadata": {
                    "manager_model": args.manager_model,
                    "manager_adapter": args.manager_adapter,
                    "manager_top_k": args.manager_top_k,
                    "answer_top_k_per_speaker": args.answer_top_k_per_speaker,
                    "memory_size": len(memory_bank),
                    "facts_source": args.timeline,
                    "invalid_manager_turns": invalid_turns,
                    "manager_trace": str(trace_path),
                },
            })
        LOGGER.info(
            "dialogue done %d/%d id=%s answer_rows=%d invalid_turns=%d",
            dialogue_index, len(timelines), dialogue_id, len(rows), invalid_turns,
        )
    write_jsonl(args.output, rows)
    LOGGER.info("wrote %d Answer tuples to %s", len(rows), args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", required=True, help="Stage-0 cached-fact timeline JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--manager-adapter", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k-per-speaker", type=int, default=30)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    build(args)


if __name__ == "__main__":
    main()
