from __future__ import annotations

from .utils import exact_match


def answer_exact_match_reward(prediction: str, gold: str) -> float:
    return exact_match(prediction, gold)


def safe_reward(prediction: str, gold: str, parse_ok: bool = True) -> float:
    if not parse_ok:
        return -0.25
    return answer_exact_match_reward(prediction, gold)
