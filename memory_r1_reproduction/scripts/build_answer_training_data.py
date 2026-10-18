"""Build Answer tuples with the baseline full-dialogue replay protocol."""

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
from memfactory.modules.memory_updater import build_manager_input
from memfactory.oracle_reward import gold_text


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
    tuple_row: dict[str, Any], manager: Any, manager_top_k: int
) -> tuple[list[dict[str, Any]], int]:
    """Replay the supplied complete dialogue from an empty bank using cached facts."""
    bank: list[dict[str, Any]] = []
    invalid_actions = 0
    for turn in tuple_row["dialogue_turns"]:
        facts = [dict(fact) for fact in turn["facts"]
                 if isinstance(fact, dict) and str(fact.get("text", "")).strip()]
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


def build_rows(timelines, manager, manager_top_k, per_speaker_top_k, split, manager_model):
    """Replay each COMPLETE dialogue once; every QA reads its final bank.

    This is the baseline Answer-data protocol, independent of Manager training
    horizon. Validation uses a separate dialogue and the same selected Manager.
    """
    rows = []
    for timeline in timelines:
        if timeline.get("split") != split:
            raise ValueError("Timeline split does not match requested Answer split")
        if "turns" not in timeline or "questions" not in timeline:
            raise ValueError("Answer data requires full dialogue timelines, not Manager windows")
        if timeline["dialogue_id"] not in SPLIT_DIALOGUE_IDS[split]:
            raise ValueError("Dialogue outside the fixed requested split")
        replay = dict(timeline, dialogue_turns=timeline["turns"])
        bank, invalid = replay_manager_window(replay, manager, manager_top_k)
        for question in timeline["questions"]:
            rows.append({"dialogue_id": timeline["dialogue_id"], "split": split,
                "participants": timeline["participants"], "question_id": str(question["question_id"]),
                "question": question["question"], "answer": gold_text(question["answer"]),
                "category": question.get("category"),
                "retrieved_memories": retrieve_per_speaker(question["question"], bank,
                    timeline["participants"], per_speaker_top_k),
                "metadata": {"source": "full_dialogue_cached_facts_greedy",
                    "manager_model": manager_model, "turns": len(timeline["turns"]),
                    "memory_size": len(bank), "invalid_actions": invalid}})
    return rows


def main():
    import argparse
    from scripts.train_manager_grpo import TextModel
    from memfactory.progress import append_progress
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--split", required=True, choices=["train", "validation"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--manager-adapter")
    parser.add_argument("--manager-device", default="cuda")
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--per-speaker-top-k", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    args = parser.parse_args()
    model = TextModel(args.manager_model, args.manager_device, False,
                      args.manager_adapter, args.max_prompt_tokens)
    progress_log = str(Path(args.output).with_suffix(".progress.jsonl"))
    class FrozenManager:
        turn = 0
        def generate(self, bank, facts, top_k):
            self.turn += 1
            append_progress(progress_log, "turn_start", turn=self.turn, memory_size=len(bank))
            prompt = build_manager_input(retrieve(" ".join(f["text"] for f in facts), bank, top_k), facts)
            raw = model.generate(prompt, args.max_new_tokens, 0.0)[0]
            append_progress(progress_log, "turn_done", turn=self.turn)
            return raw
    timelines = read_rows(args.timeline)
    rows = build_rows(timelines, FrozenManager(), args.manager_top_k,
                      args.per_speaker_top_k, args.split, args.manager_model)
    if not rows:
        raise ValueError("No Answer QA generated")
    for row in rows:
        row["metadata"].update(timeline_sha256=file_sha256(args.timeline), manager_adapter=args.manager_adapter)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} full-dialogue Answer tuples: {args.output}", flush=True)


if __name__ == "__main__":
    main()
