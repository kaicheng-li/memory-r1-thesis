"""Official LoCoMo dialogue-level split used by Memory-R1.

The published 1:1:8 split is a split of complete dialogues, not a random
split of QA rows.  Keep the IDs explicit so a reordered JSON file cannot
silently change the train/validation/test boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SPLIT_DIALOGUE_IDS = {
    "train": ("conv-26",),
    "validation": ("conv-30",),
    "test": ("conv-41", "conv-42", "conv-43", "conv-44", "conv-47", "conv-48", "conv-49", "conv-50"),
}


def select_locomo_split(payload: Any, split: str) -> list[dict[str, Any]]:
    samples = payload if isinstance(payload, list) else [payload]
    if split not in SPLIT_DIALOGUE_IDS:
        raise ValueError(f"Unknown LoCoMo split: {split}")

    by_id = {
        str(sample.get("sample_id", sample.get("dialogue_id", ""))): sample
        for sample in samples
        if isinstance(sample, dict)
    }
    expected_ids = SPLIT_DIALOGUE_IDS[split]
    missing = [dialogue_id for dialogue_id in expected_ids if dialogue_id not in by_id]
    if missing:
        raise ValueError(
            f"LoCoMo {split} split is missing dialogue IDs: {', '.join(missing)}"
        )
    selected = [by_id[dialogue_id] for dialogue_id in expected_ids]
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
