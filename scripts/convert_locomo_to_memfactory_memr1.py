from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def speaker_role(speaker: str, participants: List[str]) -> str:
    if participants and speaker == participants[0]:
        return "user"
    return "assistant"


def memory_from_turn(dialogue_id: str, turn: Dict[str, Any], turn_index: int) -> Dict[str, Any]:
    speaker = str(turn.get("speaker", "")).strip() or "Unknown"
    text = str(turn.get("text", "")).strip()
    timestamp = str(turn.get("timestamp", "")).strip()
    key = f"{speaker} turn {turn_index}"
    if timestamp:
        key = f"{key} at {timestamp}"
    return {
        "id": f"{dialogue_id}-{turn_index}",
        "key": key,
        "value": f"{speaker}: {text}",
        "memory_type": "UserMemory",
        "tags": [speaker],
        "created_at": timestamp,
        "updated_at": timestamp,
        "user_id": dialogue_id,
        "session_id": dialogue_id,
    }


def build_rows(dialogues: List[Dict[str, Any]], history_window: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dialogue in dialogues:
        dialogue_id = str(dialogue.get("dialogue_id", "dialogue"))
        participants = [str(p) for p in dialogue.get("participants", [])]
        turns = dialogue.get("turns", [])
        questions = dialogue.get("questions", [])

        for qa_index, qa in enumerate(questions):
            turn_index = qa.get("turn_index")
            if turn_index is None:
                turn_index = len(turns) - 1
            turn_index = int(turn_index)
            if turn_index < 0 or turn_index >= len(turns):
                continue

            start = max(0, turn_index - history_window)
            context_turns = turns[start:turn_index]
            current_turn = turns[turn_index]

            context_memory = [
                memory_from_turn(dialogue_id, turn, absolute_index)
                for absolute_index, turn in enumerate(turns[start:turn_index], start=start)
                if str(turn.get("text", "")).strip()
            ]

            fact = [
                {
                    "role": speaker_role(str(current_turn.get("speaker", "")), participants),
                    "content": f"{current_turn.get('speaker', '')}: {current_turn.get('text', '')}",
                    "timestamp": str(current_turn.get("timestamp", "")),
                }
            ]

            rows.append(
                {
                    "id": f"{dialogue_id}-{qa_index}",
                    "memory": context_memory,
                    "context_memory": context_memory,
                    "fact": fact,
                    "query": str(qa.get("question", "")),
                    "answer": str(qa.get("answer", "")),
                    "metadata": {
                        "dialogue_id": dialogue_id,
                        "turn_index": turn_index,
                        "participants": participants,
                        "context_turn_count": len(context_turns),
                    },
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LoCoMo-style dialogue QA data to MemFactory memory_bank data."
    )
    parser.add_argument("--input", required=True, help="Input dialogue JSON file.")
    parser.add_argument("--output", required=True, help="Output MemFactory JSONL file.")
    parser.add_argument("--history_window", type=int, default=50)
    args = parser.parse_args()

    dialogues = load_json(Path(args.input))
    rows = build_rows(dialogues, history_window=args.history_window)
    count = dump_jsonl(Path(args.output), rows)
    print(f"wrote {count} MemFactory MemR1 rows to {args.output}")


if __name__ == "__main__":
    main()
