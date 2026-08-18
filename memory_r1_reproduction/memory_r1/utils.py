from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, TypeVar


T = TypeVar("T")


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def dump_jsonl(path: str, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\"'`.,!?;:()\[\]{}]", "", text)
    return text


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0


def tokenize_for_retrieval(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def batched(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
