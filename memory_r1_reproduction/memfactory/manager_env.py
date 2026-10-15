"""Strict Memory Manager action validation and state transition semantics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


ALLOWED_EVENTS = {"ADD", "UPDATE", "DELETE", "NONE"}


@dataclass(frozen=True)
class ActionResult:
    valid: bool
    memory_bank: list[dict[str, Any]]
    applied_actions: int
    error: str | None = None


def validate_and_apply(
    memory_bank: list[dict[str, Any]],
    visible_memories: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    dialogue_id: str,
    source_turn: int,
    speaker: str,
    timestamp: str,
    max_actions: int = 5,
) -> ActionResult:
    """Validate one sampled action list, then apply it atomically.

    UPDATE/DELETE/NONE may only reference IDs displayed in the prompt. NONE
    entries are accepted (the primary Mem0 prompt instructs the model to echo
    unchanged memories with event NONE) and apply no mutation; they do not
    count as applied actions. Invalid output never partially mutates the
    bank. An empty list and an all-NONE list are both valid no-ops.
    """
    original = deepcopy(memory_bank)
    if not isinstance(decisions, list):
        return ActionResult(False, original, 0, "memory must be a list")
    mutating = sum(
        1 for decision in decisions
        if isinstance(decision, dict)
        and str(decision.get("event", "")).upper() != "NONE"
    )
    if mutating > max_actions:
        return ActionResult(
            False, original, 0, f"more than {max_actions} mutating actions"
        )
    if not decisions:
        return ActionResult(True, original, 0)

    visible_ids = {str(item.get("id", "")) for item in visible_memories}
    seen_targets: set[str] = set()
    normalized: list[tuple[str, str, str]] = []
    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            return ActionResult(False, original, 0, f"action {index} is not an object")
        event = str(decision.get("event", "")).upper()
        memory_id = str(decision.get("id", ""))
        text = str(decision.get("text", "")).strip()
        if event not in ALLOWED_EVENTS:
            return ActionResult(False, original, 0, f"unsupported event {event!r}")
        if event == "ADD":
            if not text:
                return ActionResult(False, original, 0, "ADD requires non-empty text")
        else:
            if not memory_id or memory_id not in visible_ids:
                return ActionResult(
                    False, original, 0,
                    f"{event} id {memory_id!r} was not visible in the prompt",
                )
            if memory_id in seen_targets:
                return ActionResult(False, original, 0, f"duplicate target id {memory_id!r}")
            if event == "UPDATE" and not text:
                return ActionResult(False, original, 0, "UPDATE requires non-empty text")
            seen_targets.add(memory_id)
        normalized.append((event, memory_id, text))

    bank = deepcopy(memory_bank)
    by_id = {str(item.get("id", "")): item for item in bank}
    numeric_ids = []
    for memory_id in by_id:
        prefix = f"{dialogue_id}:m"
        if memory_id.startswith(prefix) and memory_id[len(prefix):].isdigit():
            numeric_ids.append(int(memory_id[len(prefix):]))
    # Reserve a deterministic range per source turn so deleting the current
    # highest entry cannot cause a later ADD to reuse its historical ID.
    next_id = max(max(numeric_ids, default=-1) + 1, source_turn * max_actions)

    applied = 0
    for event, memory_id, text in normalized:
        if event == "NONE":
            continue
        applied += 1
        if event == "ADD":
            entry = {
                "id": f"{dialogue_id}:m{next_id}",
                "text": text,
                "speaker": speaker,
                "timestamp": timestamp,
                "source_turn": source_turn,
            }
            next_id += 1
            bank.append(entry)
            by_id[entry["id"]] = entry
        elif event == "UPDATE":
            entry = by_id[memory_id]
            entry.update({
                "text": text,
                "timestamp": timestamp,
                "source_turn": source_turn,
            })
        else:
            entry = by_id.pop(memory_id)
            bank.remove(entry)
    return ActionResult(True, bank, applied)
