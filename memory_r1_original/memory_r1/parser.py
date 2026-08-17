from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from .schemas import AnswerTrace, MemoryDecision, MemoryEvent


def _find_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    return json.loads(match.group(0))


def parse_fact_output(text: str) -> List[str]:
    try:
        data = _find_json_object(text)
        facts = data.get("facts", [])
        return [str(f).strip() for f in facts if str(f).strip()]
    except Exception:
        lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
        return lines[:4]


def parse_manager_output(text: str) -> List[MemoryDecision]:
    data = _find_json_object(text)
    decisions = []
    for item in data.get("memory", []):
        event = MemoryEvent(str(item.get("event", "NONE")).upper())
        decisions.append(
            MemoryDecision(
                id=str(item.get("id", "")),
                text=str(item.get("text", "")),
                event=event,
                speaker=str(item.get("speaker", "")),
                old_memory=str(item.get("old_memory", "")),
            )
        )
    return decisions


def parse_answer_output(text: str) -> AnswerTrace:
    try:
        data = _find_json_object(text)
        ids = [str(x) for x in data.get("selected_memory_ids", [])]
        answer = str(data.get("answer", "")).strip()
        return AnswerTrace(selected_memory_ids=ids, answer=answer, raw_text=text)
    except Exception:
        answer = text.strip().splitlines()[-1].strip()
        return AnswerTrace(selected_memory_ids=[], answer=answer, raw_text=text)
