"""Build Memory Manager training tuples directly from raw LoCoMo JSON.

For every turn t, NVIDIA NIM summarizes the preceding 50 turns into a
temporal memory bank and extracts the turn's key facts (LLMExtract,
Algorithm 5 line 7). The output row contains the bank, the window turns
(preceding 50 + the current turn, each carrying its extracted facts, replayed
by Algorithm 5), the current turn, and QA pairs linked to that turn. It
contains no memory-operation labels.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.modules.memory_extractor import build_extract_input
from memfactory.modules.memory_updater import build_manager_input

HISTORY_WINDOW = 24
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_RETRIES = 2
COMPACT_JSON_INSTRUCTION = """

IMPORTANT: Your prior response was too long or incomplete. Return ONLY one
complete JSON object. Consolidate duplicate or related facts into concise,
self-contained memories. Keep every memory text under 180 characters and the
entire response under 6000 characters. Do not repeat source dialogue or add
explanations.
"""
LOGGER = logging.getLogger("memory_r1.data_builder")


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


class NvidiaMemoryBankBuilder:
    def __init__(self, cache_path: str, model: str, max_tokens: int, max_retries: int) -> None:
        self.cache_path = Path(cache_path)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.cache = read_json(cache_path) if self.cache_path.exists() else {}
        self._facts_by_turn = {}
        self._memory_banks_by_turn = {}
        for cached_key, value in self.cache.items():
            if cached_key.startswith("facts:"):
                record_key, separator, _ = cached_key.rpartition(":")
                if separator:
                    self._facts_by_turn.setdefault(record_key, value)
            elif ":w" in cached_key:
                record_key, _, _ = cached_key.rpartition(":w")
                if record_key:
                    self._memory_banks_by_turn.setdefault(record_key, value)
        LOGGER.info(
            "teacher=%s cache=%s cached_items=%d max_tokens=%d max_retries=%d",
            model,
            cache_path,
            len(self.cache),
            max_tokens,
            max_retries,
        )

    def _facts_record(self, key: str) -> Any | None:
        return self.cache.get(key, self._facts_by_turn.get(key))

    def _memory_bank_record(self, dialogue_id: str, turn_index: int, key: str) -> Any | None:
        return self.cache.get(key, self._memory_banks_by_turn.get(f"{dialogue_id}:{turn_index}"))

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to call NVIDIA NIM.") from exc

        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set.")

        return OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
        )

    def _parse_json_content(self, content: str, label: str) -> dict[str, Any]:
        if "```json" in content:
            LOGGER.debug("JSON code fence detected: %s", label)
            content = content.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in content:
            LOGGER.debug("Generic code fence detected: %s", label)
            content = content.split("```", 1)[1].split("```", 1)[0]
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end < start:
            LOGGER.error("JSON object missing or truncated: %s response_tail=%r", label, content[-500:])
            raise ValueError("NVIDIA teacher response does not contain a JSON object.")
        json_text = content[start : end + 1]
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            LOGGER.error("JSON parse failed: %s response_tail=%r", label, json_text[-500:])
            raise
        if not isinstance(payload, dict):
            raise ValueError("NVIDIA teacher response JSON must be an object.")
        return payload

    def _json_completion(self, prompt: str, label: str) -> dict[str, Any]:
        """Call NVIDIA NIM and retry truncated or malformed JSON responses."""
        client = self._client()
        for attempt in range(self.max_retries + 1):
            retry_prompt = prompt if attempt == 0 else prompt + COMPACT_JSON_INSTRUCTION
            LOGGER.info(
                "NVIDIA request start: %s attempt=%d/%d prompt_chars=%d",
                label,
                attempt + 1,
                self.max_retries + 1,
                len(retry_prompt),
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": retry_prompt}],
                temperature=0,
                top_p=1,
                max_tokens=self.max_tokens,
                stream=False,
            )
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            # gpt-oss may expose chain-of-thought in reasoning_content. The
            # builder parses only the final answer in message.content.
            content = choice.message.content or ""
            if finish_reason == "length":
                LOGGER.warning(
                    "NVIDIA response truncated: %s attempt=%d/%d content_chars=%d tail=%r",
                    label,
                    attempt + 1,
                    self.max_retries + 1,
                    len(content),
                    content[-300:],
                )
                error: Exception = ValueError("NVIDIA response reached max_tokens.")
            elif not content.strip():
                LOGGER.warning(
                    "NVIDIA response empty: %s attempt=%d/%d finish_reason=%s reasoning_chars=%d",
                    label,
                    attempt + 1,
                    self.max_retries + 1,
                    finish_reason,
                    len(getattr(choice.message, "reasoning_content", "") or ""),
                )
                error = ValueError("NVIDIA teacher returned empty final content.")
            else:
                LOGGER.info(
                    "NVIDIA request done: %s attempt=%d/%d finish_reason=%s content_chars=%d",
                    label,
                    attempt + 1,
                    self.max_retries + 1,
                    finish_reason,
                    len(content),
                )
                try:
                    return self._parse_json_content(content, label)
                except (ValueError, json.JSONDecodeError) as exc:
                    error = exc

            if attempt < self.max_retries:
                LOGGER.warning(
                    "NVIDIA request retrying with compact output: %s reason=%s",
                    label,
                    error,
                )
                continue
            raise error

        raise AssertionError("NVIDIA retry loop exited unexpectedly.")

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary_path.replace(self.cache_path)

    def extract_facts(self, dialogue_id: str, turn_index: int, turn: dict[str, Any]) -> list[dict[str, Any]]:
        """LLMExtract(di): NVIDIA NIM extracts the turn's memory-relevant facts.

        Extraction is deterministic per turn and cached, so Algorithm 5 can
        consume the extracted facts across all rollout trajectories without
        re-running the teacher.
        """
        key = f"facts:{dialogue_id}:{turn_index}"
        record = self._facts_record(key)
        if record is not None:
            LOGGER.info("cache hit: facts dialogue=%s turn=%d", dialogue_id, turn_index)
            return record

        payload = self._json_completion(
            build_extract_input(turn),
            f"facts dialogue={dialogue_id} turn={turn_index}",
        )
        raw_facts = payload.get("memory_list", [])
        if not isinstance(raw_facts, list):
            raise ValueError(f"Invalid facts returned for {dialogue_id} turn {turn_index}.")

        facts = [
            {
                "speaker": str(turn.get("speaker", "")),
                "timestamp": str(turn.get("timestamp", "")),
                "key": str(fact.get("key", "")),
                "memory_type": str(fact.get("memory_type", "")),
                "tags": fact.get("tags", []),
                "text": str(fact.get("value", "")).strip(),
            }
            for fact in raw_facts
            if isinstance(fact, dict) and str(fact.get("value", "")).strip()
        ]

        self.cache[key] = facts
        self._save_cache()
        return facts

    def build(self, dialogue_id: str, turn_index: int, previous_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        key = f"{dialogue_id}:{turn_index}:w{HISTORY_WINDOW}"
        record = self._memory_bank_record(dialogue_id, turn_index, key)
        if record is not None:
            LOGGER.info("cache hit: memory_bank dialogue=%s turn=%d", dialogue_id, turn_index)
            return record

        retrieved_facts = [
            {
                "speaker": str(turn.get("speaker", "Unknown")),
                "timestamp": str(turn.get("timestamp", "")),
                "text": str(turn.get("text", "")).strip(),
            }
            for turn in previous_turns
            if str(turn.get("text", "")).strip()
        ]
        payload = self._json_completion(
            build_manager_input([], retrieved_facts),
            f"memory_bank dialogue={dialogue_id} turn={turn_index}",
        )
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
        self._save_cache()
        return memories


def build(input_path: str, output_path: str, cache_path: str, model: str, max_tokens: int, max_retries: int) -> None:
    dialogues = load_dialogues(read_json(input_path))
    builder = NvidiaMemoryBankBuilder(cache_path, model, max_tokens, max_retries)
    rows = []
    LOGGER.info("loaded input=%s dialogues=%d", input_path, len(dialogues))

    for dialogue_index, dialogue in enumerate(dialogues, start=1):
        validate_questions(dialogue)
        dialogue_id = str(dialogue["dialogue_id"])
        turns = dialogue["turns"]
        LOGGER.info(
            "dialogue start %d/%d id=%s turns=%d questions=%d",
            dialogue_index,
            len(dialogues),
            dialogue_id,
            len(turns),
            len(dialogue.get("questions", [])),
        )
        for turn_index, turn in enumerate(turns):
            LOGGER.info("facts progress dialogue=%s turn=%d/%d", dialogue_id, turn_index + 1, len(turns))
            turn["facts"] = (
                builder.extract_facts(dialogue_id, turn_index, turn)
                if str(turn.get("text", "")).strip()
                else []
            )
        for turn_index, current_turn in enumerate(turns):
            LOGGER.info("memory progress dialogue=%s turn=%d/%d", dialogue_id, turn_index + 1, len(turns))
            start = max(0, turn_index - HISTORY_WINDOW)
            memory_bank = builder.build(dialogue_id, turn_index, turns[start:turn_index])
            linked_questions = [
                question
                for question in dialogue.get("questions", [])
                if int(question["turn_index"]) == turn_index
            ]
            window_turns = []
            for window_index in range(start, turn_index + 1):
                window_turn = dict(turns[window_index])
                window_turn["turn_index"] = window_index
                window_turns.append(window_turn)

            rows.append(
                {
                    "dialogue_id": dialogue_id,
                    "turn_index": turn_index,
                    "participants": dialogue.get("participants", []),
                    "temporal_memory_bank": memory_bank,
                    "dialogue_turns": window_turns,
                    "current_turn": current_turn,
                    "linked_questions": linked_questions,
                    "metadata": {
                        "memory_builder": model,
                        "fact_extractor": model,
                        "history_window": HISTORY_WINDOW,
                    },
                }
            )
        LOGGER.info("dialogue done id=%s tuples=%d", dialogue_id, len(turns))

    write_jsonl(output_path, rows)
    LOGGER.info("wrote %d Manager tuples to %s", len(rows), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Memory Manager training data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--log-file", help="Optional path for a persistent build log")
    args = parser.parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")
    build(args.input, args.output, args.cache, args.model, args.max_tokens, args.max_retries)


if __name__ == "__main__":
    main()
