"""Official LoCoMo dialogue split used by Memory-R1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SPLIT_SLICES = {
    "train": slice(0, 1),
    "validation": slice(1, 2),
    "test": slice(2, 10),
}


def select_locomo_split(payload: Any, split: str) -> list[dict[str, Any]]:
    samples = payload if isinstance(payload, list) else [payload]
    selected = samples[SPLIT_SLICES[split]]
    result = []
    for sample in selected:
        row = dict(sample)
        row["qa"] = [
            question
            for question in sample.get("qa", [])
            if int(question.get("category", 0)) != 5
        ]
        result.append(row)
    return result


def question_keys(path: str, split: str) -> set[tuple[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = set()
    for sample in select_locomo_split(payload, split):
        dialogue_id = str(sample.get("sample_id", sample.get("dialogue_id", "dialogue")))
        for index, question in enumerate(sample.get("qa", [])):
            keys.add((dialogue_id, str(question.get("question_id", index))))
    return keys
