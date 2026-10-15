"""Auxiliary layered answer matching for LoCoMo diagnostics and prechecks.

This binary matcher is not the Manager or Answer RL reward. Both trainers use
the continuous token F1 in :mod:`evaluation.metrics`, following Memory-R2's
open implementation. Strict EM is retained as a separate audit comparator.
The layered rule below is used only where a deterministic binary transition is
useful, such as the Stage-0.5 go/no-go diagnostic.

Tiers (first hit wins, reward = 1):
  T1 strict   normalized string equality
  T2 date     gold's canonical dates all appear in prediction (pred <= 8 tokens)
  T3 list     every gold list item appears as a token subsequence in pred
              (gold has >= 2 items; pred <= 50 tokens)
  T4 contain  short gold (<= 6 tokens) appears as a token subsequence in pred
              (pred <= max(12, 3 * gold) tokens)
  T5 f1       token F1 >= 0.6

Guards exist to keep the relaxed reward non-gameable for Answer-Agent RL
(dumping the whole memory bank fails T2/T4 length caps and T5's precision).
"""

from __future__ import annotations

import re
from typing import Any

from evaluation.metrics import exact_match, f1_score

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles, digits for number words/ordinals."""
    text = str(text).lower()
    text = text.replace(" ", " ").replace(" ", " ")
    text = _ORDINAL.sub(r"\1", text)
    text = _NON_ALNUM.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    tokens = [_NUMBER_WORDS.get(token, token) for token in text.split()]
    return " ".join(tokens)


def tokens_of(text: str) -> list[str]:
    return normalize(text).split()


_NUMERIC_DATE = re.compile(r"\b((?:19|20)\d{2})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?\b")


def extract_dates(text: str) -> set[str]:
    """Canonical date facts: 'Y-M-D', 'Y-M', 'Y', and durations like '4 year'.

    Works on normalized token streams, so '7 May 2023', 'May 7th, 2023' and
    '2023-05-07' all yield {'2023-5-7'}.
    """
    found: set[str] = set()
    numeric_spans = []
    for match in _NUMERIC_DATE.finditer(str(text)):
        year, month, day = match.group(1), int(match.group(2)), match.group(3)
        if 1 <= month <= 12:
            found.add(f"{year}-{month}-{int(day)}" if day else f"{year}-{month}")
            numeric_spans.append(match.group(0))
    cleaned = str(text)
    for span in numeric_spans:
        cleaned = cleaned.replace(span, " ")
    tokens = tokens_of(cleaned)
    consumed = [False] * len(tokens)
    for index, token in enumerate(tokens):
        if token not in _MONTHS:
            continue
        month = _MONTHS[token]
        day = None
        year = None
        for offset in (-1, 1, 2):
            j = index + offset
            if 0 <= j < len(tokens) and tokens[j].isdigit():
                value = int(tokens[j])
                if 1 <= value <= 31 and day is None and len(tokens[j]) <= 2:
                    day = value
                    consumed[j] = True
                elif 1900 <= value <= 2100 and year is None:
                    year = value
                    consumed[j] = True
        consumed[index] = True
        if year is not None and day is not None:
            found.add(f"{year}-{month}-{day}")
        elif year is not None:
            found.add(f"{year}-{month}")
        else:
            found.add(f"m{month}" + (f"-{day}" if day is not None else ""))
    for index, token in enumerate(tokens):
        if token.isdigit() and not consumed[index] and 1900 <= int(token) <= 2100:
            found.add(token)
    for match in re.finditer(r"\b(\d+) (year|month|week|day|hour)s?\b",
                             " ".join(tokens)):
        found.add(f"{match.group(1)} {match.group(2)}")
    return found


def _parse_date(date: str) -> tuple[str | None, str | None, str | None] | None:
    """Canonical date string -> (year, month, day) with None for unspecified."""
    if re.fullmatch(r"\d+ (year|month|week|day|hour)", date):
        return None
    if date.startswith("m"):
        parts = date[1:].split("-")
        return (None, parts[0], parts[1] if len(parts) > 1 else None)
    parts = date.split("-")
    if not parts[0].isdigit():
        return None
    return (parts[0],
            parts[1] if len(parts) > 1 else None,
            parts[2] if len(parts) > 2 else None)


def dates_conflict(prediction: str, gold: str) -> bool:
    """True when both sides state dates and every pair disagrees on some
    field that both specify (e.g. pred '8 May 2023' vs gold '7 May 2023').
    Coarser statements ('May 2023' vs '7 May 2023') are compatible."""
    gold_dates = [d for d in (_parse_date(x) for x in extract_dates(gold)) if d]
    pred_dates = [d for d in (_parse_date(x) for x in extract_dates(prediction)) if d]
    if not gold_dates or not pred_dates:
        return False
    for g in gold_dates:
        for p in pred_dates:
            if all(gf == pf for gf, pf in zip(g, p)
                   if gf is not None and pf is not None):
                return False  # at least one compatible pair
    return True


def _contains_subsequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return False
    limit = len(haystack) - len(needle)
    return any(haystack[i:i + len(needle)] == needle for i in range(limit + 1))


def split_gold_items(gold: str) -> list[str]:
    parts = re.split(r"[,;]| and ", str(gold))
    items = [part.strip() for part in parts if normalize(part).strip()]
    return items


def em_strict(prediction: str, gold: str) -> float:
    """Canonical project EM, kept only as a diagnostic comparator."""
    return exact_match(prediction, gold)


def match_locomo_v2(prediction: str, gold: str,
                    f1_threshold: float = 0.6) -> dict[str, Any]:
    """Layered deterministic match. Returns reward, tier, and diagnostics."""
    pred_tokens = tokens_of(prediction)
    gold_tokens = tokens_of(gold)
    f1 = f1_score(prediction, gold)
    strict = em_strict(prediction, gold)
    result = {"reward": 0.0, "tier": "none", "em_strict": strict, "f1": f1}
    if not pred_tokens or not gold_tokens:
        return result

    if strict:
        result.update(reward=1.0, tier="strict")
        return result

    gold_dates = extract_dates(gold)
    if gold_dates and len(pred_tokens) <= 8:
        if gold_dates <= extract_dates(prediction):
            result.update(reward=1.0, tier="date")
            return result

    items = split_gold_items(gold)
    if len(items) >= 2 and len(pred_tokens) <= 50:
        if all(_contains_subsequence(pred_tokens, tokens_of(item))
               for item in items):
            result.update(reward=1.0, tier="list")
            return result

    if len(gold_tokens) <= 6 and len(pred_tokens) <= max(12, 3 * len(gold_tokens)):
        if _contains_subsequence(pred_tokens, gold_tokens):
            result.update(reward=1.0, tier="contain")
            return result

    if f1 >= f1_threshold and not dates_conflict(prediction, gold):
        result.update(reward=1.0, tier="f1")
        return result

    return result


def reward_locomo_v2(prediction: str, gold: str) -> float:
    """Auxiliary binary match score; do not use as the RL reward."""
    return match_locomo_v2(prediction, gold)["reward"]
