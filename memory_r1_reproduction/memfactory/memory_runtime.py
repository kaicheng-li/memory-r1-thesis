"""Shared runtime semantics for Memory-R1 training and inference."""

from __future__ import annotations

import json
import math
import re
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
    start, end = text.find("{"), text.rfind("}")
    payload = json.loads(text[start : end + 1])
    return payload["memory"]


def next_memory_id(memory_bank: list[dict[str, Any]], dialogue_id: str) -> str:
    prefix = f"{dialogue_id}:m"
    indices = [int(entry["id"][len(prefix) :]) for entry in memory_bank if entry["id"].startswith(prefix)]
    return f"{prefix}{max(indices, default=-1) + 1}"


def apply_decisions(
    memory_bank: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    dialogue_id: str,
    turn: dict[str, Any],
) -> None:
    by_id = {entry["id"]: entry for entry in memory_bank}
    source_turn_id = turn["dia_id"]

    for decision in decisions:
        event = decision["event"].upper()
        memory_id = str(decision.get("id", ""))
        text = decision.get("text", "").strip()

        if event == "ADD" and text:
            entry = {
                "id": next_memory_id(memory_bank, dialogue_id),
                "text": text,
                "speaker": turn["speaker"],
                "timestamp": turn["timestamp"],
                "source_turn_ids": [source_turn_id],
            }
            memory_bank.append(entry)
            by_id[entry["id"]] = entry
        elif event == "UPDATE" and memory_id in by_id and text:
            entry = by_id[memory_id]
            entry["text"] = text
            entry["timestamp"] = turn["timestamp"]
            entry["source_turn_ids"] = list(dict.fromkeys(entry["source_turn_ids"] + [source_turn_id]))
        elif event == "DELETE" and memory_id in by_id:
            memory_bank.remove(by_id.pop(memory_id))
