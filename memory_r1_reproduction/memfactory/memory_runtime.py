"""Shared runtime semantics for Memory-R1 training and inference."""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any
from memfactory.retrieval import retrieval_config, semantic_retrieve


def retrieve(query: str, memory_bank: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    if retrieval_config()["backend"] == "minilm":
        return semantic_retrieve(query, memory_bank, top_k)
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
    """Extract Manager JSON, allowing only a missing outer closing brace.

    A greedy ``first {``/``last }`` slice is vulnerable to explanations,
    multiple JSON objects, and braces inside strings. Decode the entire memory
    array before recovering its missing wrapper; never complete action content.
    Once a broken Manager object is recognized, do not fall through to echoed
    prompt examples later in the output.
    """
    decoder = json.JSONDecoder()
    error = "Memory Manager output does not contain a valid memory JSON object."
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            prefix = re.match(r'\{\s*"memory"\s*:\s*', text[start:])
            if prefix is None:
                continue
            try:
                memory, end = decoder.raw_decode(text, start + prefix.end())
            except json.JSONDecodeError:
                raise ValueError(error) from None
            tail = text[end:].lstrip()
            # A complete list plus prose/markdown may lack just its wrapper.
            # Structural continuations indicate a different syntax error.
            if not isinstance(memory, list) or (tail and tail[0] in ',]}:"[{'):
                raise ValueError(error)
            payload = {"memory": memory}
        if isinstance(payload, dict) and isinstance(payload.get("memory"), list):
            return [item for item in payload["memory"] if isinstance(item, dict)]
    raise ValueError(error)


def next_memory_id(memory_bank: list[dict[str, Any]], dialogue_id: str) -> str:
    prefix = f"{dialogue_id}:m"
    indices = []
    for entry in memory_bank:
        memory_id = str(entry.get("id", ""))
        suffix = memory_id[len(prefix) :]
        if memory_id.startswith(prefix) and suffix.isdigit():
            indices.append(int(suffix))
    return f"{prefix}{max(indices, default=-1) + 1}"


def apply_decisions(memory_bank, decisions, dialogue_id, turn, *, visible_memory_ids=None, max_actions=5):
    """Baseline transition, with a private turn-origin ledger for coverage only.

    The ledger is never shown to the policy. UPDATE retains previous origins;
    it does not assert semantic support for the current turn.
    """
    from memfactory.manager_env import validate_and_apply
    visible = memory_bank if visible_memory_ids is None else [
        m for m in memory_bank if m["id"] in visible_memory_ids]
    prior_ids = {m["id"] for m in memory_bank}
    result = validate_and_apply(
        memory_bank, visible, decisions, dialogue_id=dialogue_id,
        source_turn=int(turn.get("turn_index", 0)), speaker=turn.get("speaker", ""),
        timestamp=turn.get("timestamp", ""), max_actions=max_actions)
    if not result.valid:
        raise ValueError(result.error)
    for entry in result.memory_bank:
        if entry["id"] not in prior_ids:
            origin = str(turn.get("dia_id", ""))
            entry["source_turn_ids"] = [origin] if origin else []
    memory_bank[:] = result.memory_bank
