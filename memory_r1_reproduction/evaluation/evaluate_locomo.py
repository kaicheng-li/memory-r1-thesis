"""Formal cached-fact LoCoMo roll-forward evaluation for selected checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.audit_checkpoints import AuditModel, extract_answer
from evaluation.manager_validation import evaluate_roll_forward, read_json_or_jsonl
from memfactory.modules.memory_retriever import build_answer_input
from scripts.construct_memory_bank import top_k_per_speaker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--manager-adapter", required=True)
    parser.add_argument("--answer-model", required=True)
    parser.add_argument("--answer-adapter", required=True)
    parser.add_argument("--output", default="evaluation/results/locomo_test.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k-per-speaker", type=int, default=30)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timelines = read_json_or_jsonl(args.timeline)
    expected_qa = sum(len(item.get("questions", [])) for item in timelines)
    if len(timelines) != 8 or expected_qa != 1307:
        parser.error(
            f"formal test contract requires 8 dialogues / 1307 QA, got "
            f"{len(timelines)} / {expected_qa}"
        )
    manager = AuditModel(args.manager_model, args.manager_adapter, args.device)
    answer = AuditModel(args.answer_model, args.answer_adapter, args.device)

    def manager_generate(prompt: str) -> str:
        return manager.generate(
            prompt, args.max_prompt_tokens, args.max_new_tokens,
            temperature=0.0, seed=args.seed,
        )[0]

    def answer_generate(
        question: str, bank: list[dict[str, Any]], participants: list[str]
    ) -> str:
        retrieved = top_k_per_speaker(
            question, bank, participants, args.answer_top_k_per_speaker
        )
        memories = {
            participant: [item for item in retrieved if item["speaker"] == participant]
            for participant in participants
        }
        prompt = build_answer_input(question, memories)
        raw = answer.generate(
            prompt, args.max_prompt_tokens, args.max_new_tokens,
            temperature=0.0, seed=args.seed,
        )[0]
        return extract_answer(raw)

    result = evaluate_roll_forward(
        timelines, manager_generate, answer_generate, args.manager_top_k
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result["records"]),
        encoding="utf-8",
    )
    output.with_suffix(".json").write_text(
        json.dumps({
            "component": "answer",
            "split": "test",
            "manager_adapter": args.manager_adapter,
            "answer_adapter": args.answer_adapter,
            **result["summary"],
            "dialogue_stats": result["dialogues"],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"records: {output}")


if __name__ == "__main__":
    main()
