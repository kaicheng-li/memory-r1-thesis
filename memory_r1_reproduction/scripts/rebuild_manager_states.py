"""Stage 0: rebuild single-step Memory Manager training states (v3/v4 final plan).

Produces QA-conditioned single-step states per the paper's Section 3.1 /
Eq. (1) / Algorithm 1 reading:

    state  = (old_bank, current_facts)
    old_bank      = provenance-preserving union of all teacher-extracted facts
                    from dialogue turns STRICTLY BEFORE the target turn
                    (no current-turn facts, no future facts -> no leakage)
    current_facts = teacher facts of the target turn
    target turn   = chronologically LAST evidence turn of the QA

Hard rules (frozen in review):
  * question_id is fixed on the RAW qa-array index BEFORE category-5
    filtering; downstream must never re-enumerate.
  * evidence IDs are normalized (zero-padded turn numbers like 'D30:05'
    become 'D30:5') before mapping; QAs with no evidence vs. unmappable
    evidence are counted separately and never silently dropped.
  * bank entries carry id ('<dialogue>:m<N>', chronological), text, speaker,
    timestamp, source_turn (original dialogue turn index).

Inputs:  raw LoCoMo JSON + existing Algorithm-1 JSONL (reused as the cached
teacher-facts store; no new teacher/API calls).
Outputs: one JSONL per split with single-step states, plus a stats JSON.

Acceptance (must match exactly, see project review):
  train conv-26: 150 instances, 113 unique target turns, 0 empty-fact targets
  val   conv-30:  81 instances,  66 unique target turns, 2 empty-fact targets
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.locomo_split import questions_with_raw_ids

SPLIT_DIALOGUE_IDS = {
    "train": ("conv-26",),
    "validation": ("conv-30",),
    "test": ("conv-41", "conv-42", "conv-43", "conv-44",
             "conv-47", "conv-48", "conv-49", "conv-50"),
}

EVIDENCE_PATTERN = re.compile(r"\bD(\d+):(\d+)\b")


def normalize_evidence(value: Any) -> list[str]:
    """Extract and canonicalize evidence IDs ('D30:05' -> 'D30:5')."""
    items = value if isinstance(value, list) else [value]
    ids = []
    for item in items:
        for session, turn in EVIDENCE_PATTERN.findall(str(item)):
            ids.append(f"D{int(session)}:{int(turn)}")
    return ids


def flatten_dialogue(sample: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    """Flatten LoCoMo sessions to a turn list; map canonical dia_id -> index."""
    conversation = sample.get("conversation", {})
    session_names = sorted(
        (name for name, value in conversation.items()
         if re.fullmatch(r"session_\d+", name) and isinstance(value, list)),
        key=lambda name: int(name.split("_")[1]),
    )
    turns = []
    dia_to_index = {}
    for name in session_names:
        for raw_turn in conversation[name]:
            if not isinstance(raw_turn, dict):
                continue
            index = len(turns)
            raw_id = str(raw_turn.get("dia_id", ""))
            canonical = normalize_evidence(raw_id)
            turns.append({
                "turn_index": index,
                "speaker": str(raw_turn.get("speaker", "")),
                "text": str(raw_turn.get("text", "")),
                "timestamp": str(raw_turn.get("timestamp",
                                 conversation.get(f"{name}_date_time", ""))),
                "dia_id": canonical[0] if canonical else raw_id,
            })
            if canonical:
                dia_to_index[canonical[0]] = index
    participants = [str(value) for key in ("speaker_a", "speaker_b")
                    if (value := conversation.get(key))]
    return turns, dia_to_index, participants


def load_cached_facts(jsonl_path: Path, dialogue_id: str) -> dict[int, list[dict[str, Any]]]:
    """Recover per-turn teacher facts from the existing Algorithm-1 JSONL."""
    facts_by_turn: dict[int, list[dict[str, Any]]] = {}
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("dialogue_id")) != dialogue_id:
                continue
            for turn in row.get("dialogue_turns", []):
                turn_index = int(turn.get("turn_index", -1))
                if turn_index >= 0 and turn_index not in facts_by_turn:
                    facts_by_turn[turn_index] = [
                        fact for fact in turn.get("facts", [])
                        if isinstance(fact, dict) and str(fact.get("text", "")).strip()
                    ]
    return facts_by_turn


def build_fact_bank(dialogue_id: str,
                    turns: list[dict[str, Any]],
                    facts_by_turn: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Provenance-preserving chronological fact bank over the whole dialogue."""
    bank = []
    counter = 0
    for turn in turns:
        turn_index = turn["turn_index"]
        for fact in facts_by_turn.get(turn_index, []):
            bank.append({
                "id": f"{dialogue_id}:m{counter}",
                "text": str(fact.get("text", "")).strip(),
                "speaker": str(fact.get("speaker", turn["speaker"])),
                "timestamp": str(fact.get("timestamp", turn["timestamp"])),
                "source_turn": turn_index,
                "key": str(fact.get("key", "")),
                "memory_type": str(fact.get("memory_type", "")),
                "tags": fact.get("tags", []),
            })
            counter += 1
    return bank


def rebuild_split(raw_samples: list[dict[str, Any]],
                  facts_jsonl: Path,
                  split: str,
                  build_states: bool = True,
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    instances = []
    timelines = []
    stats: dict[str, Any] = {
        "split": split,
        "dialogues": [],
        "qa_cat1_4": 0,
        "bindable": 0,
        "no_evidence": [],          # (dialogue_id, raw question_id)
        "unmappable_evidence": [],  # all evidence IDs missing from dialogue
        "partial_evidence": [],     # some evidence IDs missing (kept, bound to last mappable)
        "empty_fact_targets": [],   # target turn exists but teacher facts empty
        "unique_target_turns": {},
        "category_counts": Counter(),
    }
    for sample in raw_samples:
        dialogue_id = str(sample.get("sample_id", sample.get("dialogue_id", "")))
        if dialogue_id not in SPLIT_DIALOGUE_IDS[split]:
            continue
        turns, dia_to_index, participants = flatten_dialogue(sample)
        facts_by_turn = load_cached_facts(facts_jsonl, dialogue_id)
        missing_fact_turns = [t["turn_index"] for t in turns
                              if t["turn_index"] not in facts_by_turn and t["text"].strip()]
        if missing_fact_turns:
            raise SystemExit(
                f"{dialogue_id}: {len(missing_fact_turns)} turns lack cached teacher "
                f"facts (e.g. {missing_fact_turns[:5]}); cannot rebuild without re-running Algorithm 1."
            )
        full_bank = build_fact_bank(dialogue_id, turns, facts_by_turn)
        targets = set()

        split_questions = []
        for qa in questions_with_raw_ids(sample):
            category = int(qa.get("category", 0))
            if category == 5:          # adversarial: excluded AFTER id fixation
                continue
            question_id = str(qa["question_id"])
            stats["qa_cat1_4"] += 1
            stats["category_counts"][category] += 1

            answer = qa.get("answer", "")
            if isinstance(answer, list):
                answer = ", ".join(str(item) for item in answer)
            split_questions.append({
                "question_id": question_id,
                "question": str(qa.get("question", "")),
                "answer": str(answer),
                "category": category,
            })
            if not build_states:
                continue

            evidence = normalize_evidence(qa.get("evidence", []))
            mapped = [eid for eid in evidence if eid in dia_to_index]
            if not evidence:
                stats["no_evidence"].append((dialogue_id, question_id))
                continue
            if not mapped:
                stats["unmappable_evidence"].append((dialogue_id, question_id, evidence))
                continue
            if len(mapped) < len(evidence):
                stats["partial_evidence"].append(
                    (dialogue_id, question_id,
                     [eid for eid in evidence if eid not in dia_to_index]))

            target_turn = max(dia_to_index[eid] for eid in mapped)
            targets.add(target_turn)
            current_facts = facts_by_turn.get(target_turn, [])
            if not current_facts:
                stats["empty_fact_targets"].append((dialogue_id, question_id, target_turn))

            instances.append({
                "dialogue_id": dialogue_id,
                "split": split,
                "question_id": question_id,
                "question": str(qa.get("question", "")),
                "answer": str(answer),
                "category": category,
                "participants": participants,
                "target_turn": target_turn,
                "target_speaker": turns[target_turn]["speaker"],
                "target_timestamp": turns[target_turn]["timestamp"],
                "evidence": evidence,
                "evidence_mapped": mapped,
                "target_has_facts": bool(current_facts),
                "current_facts": current_facts,
                "old_bank": [entry for entry in full_bank
                             if entry["source_turn"] < target_turn],
            })
            stats["bindable"] += 1

        timelines.append({
            "dialogue_id": dialogue_id,
            "split": split,
            "participants": participants,
            "turns": [
                {
                    **turn,
                    "facts": facts_by_turn.get(turn["turn_index"], []),
                }
                for turn in turns
            ],
            "questions": split_questions,
        })

        stats["dialogues"].append(dialogue_id)
        stats["unique_target_turns"][dialogue_id] = len(targets)
    stats["category_counts"] = dict(stats["category_counts"])
    return instances, timelines, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/locomo10.json")
    parser.add_argument("--facts-jsonl", default="data/manager_training_data_vllm.jsonl",
                        help="Existing Algorithm-1 JSONL used as the cached teacher-facts store")
    parser.add_argument("--output-dir", default="data/stage0")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"],
                        choices=list(SPLIT_DIALOGUE_IDS))
    parser.add_argument(
        "--timeline-only", action="store_true",
        help="Write cached-fact timelines without materializing duplicated QA states",
    )
    args = parser.parse_args()

    raw_samples = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    if not isinstance(raw_samples, list):
        raw_samples = [raw_samples]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split in args.splits:
        instances, timelines, stats = rebuild_split(
            raw_samples, Path(args.facts_jsonl), split,
            build_states=not args.timeline_only,
        )
        out_path = output_dir / f"manager_states_{split}.jsonl"
        if not args.timeline_only:
            with out_path.open("w", encoding="utf-8") as handle:
                for instance in instances:
                    handle.write(json.dumps(instance, ensure_ascii=False) + "\n")
        timeline_path = output_dir / f"dialogue_timelines_{split}.json"
        timeline_path.write_text(
            json.dumps(timelines, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        all_stats[split] = stats
        if not args.timeline_only:
            print(f"[{split}] wrote {len(instances)} states -> {out_path}")
        print(f"  timelines={len(timelines)} -> {timeline_path}")
        print(f"  cat1-4={stats['qa_cat1_4']} bindable={stats['bindable']} "
              f"no_evidence={len(stats['no_evidence'])} "
              f"unmappable={len(stats['unmappable_evidence'])} "
              f"partial={len(stats['partial_evidence'])} "
              f"empty_fact_targets={len(stats['empty_fact_targets'])} "
              f"unique_targets={stats['unique_target_turns']}")
        for label in ("no_evidence", "unmappable_evidence", "partial_evidence", "empty_fact_targets"):
            for item in stats[label]:
                print(f"    {label}: {item}")

    stats_path = output_dir / "stage0_stats.json"
    merged_stats = {}
    if stats_path.is_file():
        merged_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    merged_stats.update(all_stats)
    stats_path.write_text(json.dumps(merged_stats, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()
