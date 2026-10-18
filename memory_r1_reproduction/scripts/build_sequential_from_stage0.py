"""Adapt cached Stage-0 timelines and QA bindings to sequential Manager tuples.

No old_bank snapshots are consumed. Each trajectory still starts empty.
--history-window -1 uses the complete causal prefix and retains long-span QA.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.locomo_split import SPLIT_DIALOGUE_IDS
from memfactory.run_metadata import file_sha256


def build_rows(timelines: list[dict], states: list[dict], split: str, history_window: int):
    if history_window < -1:
        raise ValueError("history_window must be -1 (full prefix) or nonnegative")
    allowed = set(SPLIT_DIALOGUE_IDS[split])
    by_dialogue = {}
    for timeline in timelines:
        dialogue_id = str(timeline["dialogue_id"])
        if dialogue_id not in allowed or timeline.get("split") != split or dialogue_id in by_dialogue:
            raise ValueError(f"Unexpected/duplicate timeline {dialogue_id}")
        by_dialogue[dialogue_id] = timeline
    if set(by_dialogue) != allowed:
        raise ValueError("Missing dialogue timeline")
    bindings, omitted, seen = {}, [], set()
    for state in states:
        dialogue_id, question_id = str(state["dialogue_id"]), str(state["question_id"])
        key = dialogue_id, question_id
        if state.get("split") != split or dialogue_id not in by_dialogue or key in seen:
            raise ValueError(f"Unexpected/duplicate QA binding {key}")
        seen.add(key)
        timeline = by_dialogue[dialogue_id]
        question_lookup = {str(q["question_id"]): q for q in timeline["questions"]}
        question = question_lookup[question_id]
        if (state["question"] != question["question"] or state["answer"] != question["answer"]
                or int(state["category"]) != int(question["category"])):
            raise ValueError(f"QA identity/content mismatch: {key}")
        if int(question["category"]) not in {1, 2, 3, 4}:
            raise ValueError(f"Unexpected category: {key}")
        turns = timeline["turns"]
        positions = {turn["dia_id"]: index for index, turn in enumerate(turns)}
        evidence = list(dict.fromkeys(state["evidence"]))
        if not evidence or any(value not in positions for value in evidence):
            raise ValueError(f"Missing/unmapped evidence: {key}")
        target = max(positions[value] for value in evidence)
        if int(state["target_turn"]) != target:
            raise ValueError(f"Target must be the final evidence turn: {key}")
        start = 0 if history_window == -1 else max(0, target - history_window)
        span = target - min(positions[value] for value in evidence)
        if min(positions[value] for value in evidence) < start:
            omitted.append({"dialogue_id": dialogue_id, "question_id": question_id, "evidence_span": span})
            continue
        linked = dict(question, evidence=evidence,
                      evidence_turn_indices=[positions[value] for value in evidence], turn_index=target)
        bindings.setdefault((dialogue_id, target), []).append(linked)
    rows = []
    for (dialogue_id, target), questions in sorted(bindings.items()):
        timeline = by_dialogue[dialogue_id]
        start = 0 if history_window == -1 else max(0, target - history_window)
        turns = timeline["turns"][start:target + 1]
        if any(not isinstance(turn.get("facts"), list) for turn in turns):
            raise ValueError(f"Missing cached facts: {dialogue_id}:{target}")
        rows.append({
            "dialogue_id": dialogue_id, "split": split, "turn_index": target,
            "participants": timeline["participants"], "temporal_memory_bank": [],
            "dialogue_turns": turns, "current_turn": timeline["turns"][target],
            "linked_questions": questions,
            "metadata": {"split": split, "source": "stage0_cached_sequential_v1",
                         "initial_bank": "empty", "history_window": history_window},
        })
    manifest = {
        "split": split, "history_window": history_window, "initial_bank": "empty",
        "source_bound_qa": len(states), "rows": len(rows),
        "linked_questions": sum(len(row["linked_questions"]) for row in rows),
        "omitted_out_of_window": omitted,
        "timeline_qa": sum(len(item["questions"]) for item in timelines),
        "fact_source": "existing_stage0_cache_not_reextracted",
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timelines", required=True)
    parser.add_argument("--states", required=True)
    parser.add_argument("--split", required=True, choices=["train", "validation"])
    parser.add_argument("--history-window", type=int, default=-1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    timelines = json.loads(Path(args.timelines).read_text(encoding="utf-8"))
    states = [json.loads(line) for line in Path(args.states).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, manifest = build_rows(timelines, states, args.split, args.history_window)
    manifest.update({"timeline_sha256": file_sha256(args.timelines), "states_sha256": file_sha256(args.states)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["output_sha256"] = file_sha256(output)
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "omitted_out_of_window"}, ensure_ascii=False))
    print(f"omitted_out_of_window={len(manifest['omitted_out_of_window'])}")


if __name__ == "__main__":
    main()
