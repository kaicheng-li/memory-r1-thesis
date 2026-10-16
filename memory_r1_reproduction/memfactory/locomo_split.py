"""Dialogue-level LoCoMo split and stable question identities.

Question IDs are assigned before filtering the adversarial category so that
the same QA keeps its identity in data construction and evaluation.
"""

from __future__ import annotations

from typing import Any


SPLIT_DIALOGUE_IDS = {
    "train": ("conv-26",),
    "validation": ("conv-30",),
    "test": (
        "conv-41", "conv-42", "conv-43", "conv-44",
        "conv-47", "conv-48", "conv-49", "conv-50",
    ),
}


def questions_with_raw_ids(sample: dict[str, Any]) -> list[dict[str, Any]]:
    questions = []
    for raw_index, question in enumerate(sample.get("qa", [])):
        if not isinstance(question, dict):
            continue
        row = dict(question)
        row["question_id"] = str(question.get("question_id", raw_index))
        questions.append(row)
    return questions


def select_locomo_split(payload: Any, split: str) -> list[dict[str, Any]]:
    """Select complete dialogues and LoCoMo categories 1--4 only."""
    if split not in SPLIT_DIALOGUE_IDS:
        raise ValueError(f"Unknown LoCoMo split: {split}")
    samples = payload if isinstance(payload, list) else [payload]
    by_id = {
        str(sample.get("sample_id", sample.get("dialogue_id", ""))): sample
        for sample in samples
        if isinstance(sample, dict)
    }
    expected_ids = SPLIT_DIALOGUE_IDS[split]
    missing = [dialogue_id for dialogue_id in expected_ids if dialogue_id not in by_id]
    if missing:
        raise ValueError(f"LoCoMo {split} split is missing dialogue IDs: {missing}")
    result = []
    for dialogue_id in expected_ids:
        sample = by_id[dialogue_id]
        row = dict(sample)
        row["qa"] = [
            question for question in questions_with_raw_ids(sample)
            if int(question.get("category", 0)) in {1, 2, 3, 4}
        ]
        result.append(row)
    return result
