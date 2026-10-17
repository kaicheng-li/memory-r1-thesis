"""Small append-only JSONL progress logger used by long local runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def append_progress(path: str | Path | None, event: str, **fields: Any) -> None:
    """Append one flushed progress event; logging failure never changes training."""
    if not path:
        return
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
    except (OSError, TypeError, ValueError):
        return
