"""Build Memory Manager training tuples directly from raw LoCoMo JSON.

For every turn t, GPT-4o-mini summarizes the preceding 50 turns into a
temporal memory bank. The output row contains the bank, the current turn, and
QA pairs linked to that turn. It contains no memory-operation labels.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_updater import build_manager_input

HISTORY_WINDOW = 50


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _session_number(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def normalize_locomo_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Flatten LoCoMo sessions and link QA evidence to turn indices."""
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, dict):
        raise ValueError("LoCoMo sample conversation must be an object.")

    turns = []
    evidence_to_index = {}
    session_names = sorted(
        (name for name, value in conversation.items() if name.startswith("session_") and isinstance(value, list)),
        key=_session_number,
    )
    for session_name in session_names:
        for local_index, raw_turn in enumerate(conversation[session_name]):
            if not isinstance(raw_turn, dict):
                continue
            turn_index = len(turns)
            dia_id = str(raw_turn.get("dia_id", f"D{_session_number(session_name)}:{local_index}"))
            turns.append({
                "speaker": str(raw_turn.get("speaker", "")),
                "text": str(raw_turn.get("text", "")),
                "timestamp": str(raw_turn.get("timestamp", conversation.get(f"{session_name}_date_time", ""))),
                "dia_id": dia_id,
            })
            evidence_to_index[dia_id] = turn_index

    questions = []
    for qa_index, raw_qa in enumerate(sample.get("qa", [])):
        if not isinstance(raw_qa, dict):
            continue
        evidence = raw_qa.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        turn_index = raw_qa.get("turn_index")
        if turn_index is None:
            turn_index = next((evidence_to_index.get(str(item)) for item in evidence if str(item) in evidence_to_index), None)
        if turn_index is None:
            continue
        answer = raw_qa.get("answer", "")
        if isinstance(answer, list):
            answer = ", ".join(str(item) for item in answer)
        questions.append({
            "question_id": str(raw_qa.get("question_id", qa_index)),
            "turn_index": int(turn_index),
            "question": str(raw_qa.get("question", "")),
            "answer": str(answer),
            "evidence": evidence,
        })

    participants = [str(value) for key in ("speaker_a", "speaker_b") if (value := conversation.get(key))]
    if not participants:
        participants = list(dict.fromkeys(turn["speaker"] for turn in turns if turn["speaker"]))
    return {
        "dialogue_id": str(sample.get("sample_id", sample.get("dialogue_id", "dialogue"))),
        "participants": participants,
        "turns": turns,
        "questions": questions,
    }


def load_dialogues(payload: Any) -> list[dict[str, Any]]:
    samples = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError("Input must be raw LoCoMo JSON object or list of objects.")
    return [normalize_locomo_sample(sample) if "conversation" in sample else sample for sample in samples]


def validate_questions(dialogue: dict[str, Any]) -> None:
    turn_count = len(dialogue.get("turns", []))
    for index, question in enumerate(dialogue.get("questions", [])):
        if "turn_index" not in question:
            raise ValueError(f"{dialogue['dialogue_id']} question {index} has no turn_index.")
        try:
            turn_index = int(question["turn_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{dialogue['dialogue_id']} question {index} has an invalid turn_index.") from exc
        if not 0 <= turn_index < turn_count:
            raise ValueError(f"{dialogue['dialogue_id']} question {index} points outside its turns.")


class GPTMemoryBankBuilder:
    def __init__(self, cache_path: str, model: str) -> None:
        self.cache_path = Path(cache_path)
        self.model = model
        self.cache = read_json(cache_path) if self.cache_path.exists() else {}

    def build(self, dialogue_id: str, turn_index: int, previous_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        key = f"{dialogue_id}:{turn_index}"
        if key in self.cache:
            return self.cache[key]

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai to build GPT temporal memory banks.") from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        retrieved_facts = [
            {
                "speaker": str(turn.get("speaker", "Unknown")),
                "timestamp": str(turn.get("timestamp", "")),
                "text": str(turn.get("text", "")).strip(),
            }
            for turn in previous_turns
            if str(turn.get("text", "")).strip()
        ]
        client_args = {"api_key": api_key}
        if os.environ.get("OPENAI_BASE_URL"):
            client_args["base_url"] = os.environ["OPENAI_BASE_URL"]
        client = OpenAI(**client_args)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": build_manager_input([], retrieved_facts),
                }
            ],
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        raw_memories = payload.get("memory", [])
        if not isinstance(raw_memories, list):
            raise ValueError(f"Invalid memory_bank returned for {dialogue_id} turn {turn_index}.")

        memories = []
        for memory_index, memory in enumerate(raw_memories):
            text = str(memory.get("text", "")).strip() if isinstance(memory, dict) else ""
            if not text:
                continue
            if str(memory.get("event", "ADD")).upper() == "DELETE":
                continue
            memories.append(
                {
                    "id": f"{dialogue_id}:t{turn_index}:m{memory_index}",
                    "text": text,
                    "timestamp": str(memory.get("timestamp", "")),
                    "source_turn": memory.get("source_turn"),
                }
            )

        self.cache[key] = memories
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return memories


def build(input_path: str, output_path: str, cache_path: str, model: str) -> None:
    dialogues = load_dialogues(read_json(input_path))
    builder = GPTMemoryBankBuilder(cache_path, model)
    rows = []

    for dialogue in dialogues:
        validate_questions(dialogue)
        dialogue_id = str(dialogue["dialogue_id"])
        turns = dialogue["turns"]
        for turn_index, current_turn in enumerate(turns):
            start = max(0, turn_index - HISTORY_WINDOW)
            memory_bank = builder.build(dialogue_id, turn_index, turns[start:turn_index])
            linked_questions = [
                question
                for question in dialogue.get("questions", [])
                if int(question["turn_index"]) == turn_index
            ]
            rows.append(
                {
                    "dialogue_id": dialogue_id,
                    "turn_index": turn_index,
                    "participants": dialogue.get("participants", []),
                    "temporal_memory_bank": memory_bank,
                    "current_turn": current_turn,
                    "linked_questions": linked_questions,
                    "metadata": {
                        "memory_builder": model,
                        "history_window": HISTORY_WINDOW,
                    },
                }
            )

    write_jsonl(output_path, rows)
    print(f"wrote {len(rows)} Manager tuples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Memory Manager training data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    build(args.input, args.output, args.cache, args.model)


if __name__ == "__main__":
    main()
