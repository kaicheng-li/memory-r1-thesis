"""Best-effort, flushed JSONL progress logging for long local runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any


def append_progress(path: str | Path | None, event: str, **fields: Any) -> None:
    """Append one immediately flushed progress event without affecting a run."""
    if not path:
        return
    record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    try:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    except Exception as exc:
        print(f"[progress] Cannot emit {event}: {exc}", file=sys.stderr, flush=True)
        return
    try:
        print(line, flush=True)
    except OSError:
        # A closed terminal/pipe must not suppress the file log.
        pass
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
    except Exception as exc:
        # Keep training alive, but never hide a missing audit trail.
        print(f"[progress] Cannot write {target.resolve()}: {exc}", file=sys.stderr, flush=True)
