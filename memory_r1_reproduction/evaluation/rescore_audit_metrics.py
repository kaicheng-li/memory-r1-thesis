"""Backfill paper metrics and LoCoMo categories into checkpoint audit files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import (
    add_question_metadata,
    aggregate_answer_metrics,
    load_question_metadata,
    mean,
    score_record,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def rescore(path: Path, metadata: dict[tuple[str, str], dict[str, Any]]) -> None:
    records = read_jsonl(path)
    for record in records:
        add_question_metadata(record, metadata)
        score_record(record)
    write_atomic(
        path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )

    summary_path = path.with_suffix(".json")
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    value_keys = ("f1", "bleu1", "llm_judge")
    paper_metrics = aggregate_answer_metrics(records, value_keys)
    summary["metric_scorer"] = "memory_r1_locomo_v1"
    summary["em" if summary.get("component") == "answer" else "em_all"] = mean(records, "em") or 0.0
    summary["f1" if summary.get("component") == "answer" else "f1_all"] = mean(records, "f1") or 0.0
    summary["bleu1" if summary.get("component") == "answer" else "bleu1_all"] = mean(records, "bleu1") or 0.0
    if summary.get("component") == "manager":
        complete = [record for record in records if record.get("trajectory_complete")]
        summary["em_complete"] = mean(complete, "em") or 0.0
        summary["f1_complete"] = mean(complete, "f1") or 0.0
        summary["bleu1_complete"] = mean(complete, "bleu1") or 0.0
        summary["paper_metrics_all"] = paper_metrics
        summary["paper_metrics_complete"] = aggregate_answer_metrics(complete, value_keys)
    else:
        summary["paper_metrics"] = paper_metrics
    if paper_metrics["overall"].get("llm_judge") is not None:
        summary["llm_judge"] = paper_metrics["overall"]["llm_judge"]
    write_atomic(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"rescored {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir")
    parser.add_argument("--raw-data-path", required=True)
    parser.add_argument("--pattern", default="*.jsonl")
    args = parser.parse_args()
    paths = sorted(Path(args.audit_dir).glob(args.pattern))
    if not paths:
        parser.error(f"no audit record files matched {args.pattern!r}")
    metadata = load_question_metadata(args.raw_data_path)
    for path in paths:
        rescore(path, metadata)


if __name__ == "__main__":
    main()
