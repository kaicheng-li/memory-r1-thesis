"""Read-only verification of the formal fullprefix QA/data contract."""
import argparse
import json
from pathlib import Path


def check_data(rows, states):
    expected = {(s["dialogue_id"], str(s["question_id"])): s for s in states}
    seen = set()
    for row in rows:
        turns = row["dialogue_turns"]
        target = int(row["turn_index"])
        if row.get("temporal_memory_bank") or row.get("initial_memory_bank"):
            raise ValueError("Formal fullprefix runs start empty")
        if [int(t["turn_index"]) for t in turns] != list(range(target + 1)):
            raise ValueError("Expected complete causal prefix 0..target, not window24")
        for q in row["linked_questions"]:
            key = row["dialogue_id"], str(q["question_id"])
            if key in seen or key not in expected:
                raise ValueError("Duplicate or unexpected QA")
            s = expected[key]
            if (q["question"], q["answer"], target, set(q["evidence"])) != (
                    s["question"], s["answer"], int(s["target_turn"]), set(s["evidence"])):
                raise ValueError(f"QA/target mismatch: {key}")
            seen.add(key)
    if seen != set(expected):
        raise ValueError("Sequential and baseline QA sets differ")
    return {"tuples": len(rows), "qa": len(seen), "raw_replay_turns": sum(len(r["dialogue_turns"]) for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--states", required=True)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = check_data(read(args.data_path), read(args.states))
    if (result["tuples"], result["qa"]) != (113, 150):
        raise ValueError("Formal LoCoMo train contract requires 113 targets / 150 QA")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
