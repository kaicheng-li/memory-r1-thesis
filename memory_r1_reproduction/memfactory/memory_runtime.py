"""Shared runtime semantics for Memory-R1 training and inference."""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any


def retrieve(query: str, memory_bank: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    query_tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))

    def score(memory: dict[str, Any]) -> float:
        memory_tokens = set(re.findall(r"[a-zA-Z0-9]+", memory["text"].lower()))
        denominator = math.sqrt(len(query_tokens) * len(memory_tokens))
        return len(query_tokens & memory_tokens) / denominator if denominator else 0.0

    return sorted(memory_bank, key=score, reverse=True)[:top_k]


def retrieve_per_speaker(
    query: str,
    memory_bank: list[dict[str, Any]],
    participants: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    return [
        memory
        for participant in participants
        for memory in retrieve(
            query,
            [entry for entry in memory_bank if entry["speaker"] == participant],
            top_k,
        )
    ]


def parse_manager_output(text: str) -> list[dict[str, Any]]:
    """Extract the first valid Manager JSON object from model output.

    A greedy ``first {``/``last }`` slice is vulnerable to explanations,
    multiple JSON objects, and braces inside strings.  ``raw_decode`` lets us
    try each possible object boundary without accepting a malformed prefix.
    """
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("memory"), list):
            return payload["memory"]
    raise ValueError("Memory Manager output does not contain a valid memory JSON object.")


def next_memory_id(memory_bank: list[dict[str, Any]], dialogue_id: str) -> str:
    prefix = f"{dialogue_id}:m"
    indices = []
    for entry in memory_bank:
        memory_id = str(entry.get("id", ""))
        suffix = memory_id[len(prefix) :]
        if memory_id.startswith(prefix) and suffix.isdigit():
            indices.append(int(suffix))
    return f"{prefix}{max(indices, default=-1) + 1}"


def apply_decisions(
    memory_bank: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    dialogue_id: str,
    turn: dict[str, Any],
    *,
    visible_memory_ids: set[str] | None = None,
    max_actions: int = 5,
) -> None:
    """Validate and apply a Manager action list atomically.

    The prompt exposes only a retrieved subset of the bank.  When that set is
    supplied, UPDATE/DELETE/NONE may reference only those IDs.  All validation
    happens before mutation, so a malformed later action cannot leave a partial
    state behind.
    """
    if not isinstance(decisions, list):
        raise ValueError("memory must be a list")
    mutating = sum(
        1 for item in decisions
        if isinstance(item, dict) and str(item.get("event", "")).upper() != "NONE"
    )
    if mutating > max_actions:
        raise ValueError(f"more than {max_actions} mutating actions")

    original = deepcopy(memory_bank)
    by_id = {str(entry.get("id", "")): entry for entry in original}
    visible_ids = (
        {str(value) for value in visible_memory_ids}
        if visible_memory_ids is not None
        else set(by_id)
    )
    seen_targets: set[str] = set()
    normalized: list[tuple[str, str, str]] = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            raise ValueError(f"action {index} is not an object")
        event = str(decision.get("event", "")).upper()
        if event not in {"ADD", "UPDATE", "DELETE", "NONE"}:
            raise ValueError(f"unsupported event {event!r}")
        memory_id = str(decision.get("id", ""))
        text = str(decision.get("text", "")).strip()
        if event == "ADD":
            if not text:
                raise ValueError("ADD requires non-empty text")
        else:
            if not memory_id or memory_id not in visible_ids or memory_id not in by_id:
                raise ValueError(f"{event} id {memory_id!r} was not visible")
            if memory_id in seen_targets:
                raise ValueError(f"duplicate target id {memory_id!r}")
            if event == "UPDATE" and not text:
                raise ValueError("UPDATE requires non-empty text")
            seen_targets.add(memory_id)
        normalized.append((event, memory_id, text))

    source_turn_id = str(turn.get("dia_id", ""))
    bank = deepcopy(original)
    by_id = {str(entry.get("id", "")): entry for entry in bank}
    for event, memory_id, text in normalized:
        if event == "NONE":
            continue
        if event == "ADD":
            entry = {
                "id": next_memory_id(bank, dialogue_id),
                "text": text,
                "speaker": turn.get("speaker", ""),
                "timestamp": turn.get("timestamp", ""),
                "source_turn_ids": [source_turn_id] if source_turn_id else [],
            }
            bank.append(entry)
            by_id[entry["id"]] = entry
        elif event == "UPDATE":
            entry = by_id[memory_id]
            entry["text"] = text
            entry["timestamp"] = turn.get("timestamp", "")
            prior_sources = [str(value) for value in entry.get("source_turn_ids", [])]
            # An UPDATE can rewrite an old fact without containing any new
            # evidence from this turn.  Preserve provenance; do not award the
            # current turn merely because an UPDATE happened here.
            entry["source_turn_ids"] = list(dict.fromkeys(prior_sources))
        elif event == "DELETE":
            entry = by_id.pop(memory_id)
            bank.remove(entry)
    memory_bank[:] = bank
