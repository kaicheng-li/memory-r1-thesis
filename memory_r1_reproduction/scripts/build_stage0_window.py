"""Derive a windowed Stage-0 variant: clip each state's old_bank to the paper's
preceding-K-turns horizon (Memory-R1 B.2 uses 24).

Reads  data/stage0/manager_states_{split}.jsonl  (full-history old_bank)
Writes <output-dir>/manager_states_{split}.jsonl (old_bank clipped to
       source_turn >= target_turn - K) plus copies of the dialogue timelines
       (window-independent) and a manifest with SHA256 provenance.

Pure CPU, no LLM calls. QA rows are never dropped: only old_bank entries are
filtered, so the 150/81 QA contracts still hold downstream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clip_split(source: Path, target: Path, window: int) -> dict:
    rows = kept = dropped = 0
    clipped_rows = 0
    with source.open(encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            target_turn = int(row["target_turn"])
            cutoff = target_turn - window
            bank = row["old_bank"]
            for entry in bank:  # contract: fields must exist and be causal
                if int(entry["source_turn"]) >= target_turn:
                    raise ValueError(
                        f"non-causal old_bank entry {entry['id']} "
                        f"(source_turn {entry['source_turn']} >= target {target_turn})"
                    )
                if not str(entry.get("speaker", "")).strip():
                    raise ValueError(f"old_bank entry {entry['id']} lacks speaker")
            filtered = [e for e in bank if int(e["source_turn"]) >= cutoff]
            rows += 1
            kept += len(filtered)
            dropped += len(bank) - len(filtered)
            if len(filtered) != len(bank):
                clipped_rows += 1
            row["old_bank"] = filtered
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"qa_rows": rows, "entries_kept": kept, "entries_dropped": dropped,
            "rows_affected": clipped_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-dir", default="data/stage0")
    parser.add_argument("--output-dir", default="data/stage0_w24")
    parser.add_argument("--window", type=int, default=24,
                        help="Keep old_bank entries with source_turn >= target_turn - window (paper B.2: 24)")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    args = parser.parse_args()
    if args.window < 1:
        parser.error("window must be positive")

    stage0 = Path(args.stage0_dir)
    output = Path(args.output_dir)
    if (output / "manifest.json").exists():
        raise SystemExit(f"{output} already contains a derived dataset; use a new directory")
    output.mkdir(parents=True, exist_ok=True)

    manifest = {"window": args.window, "source_dir": str(stage0), "splits": {}, "copies": {}}
    for split in args.splits:
        source = stage0 / f"manager_states_{split}.jsonl"
        target = output / f"manager_states_{split}.jsonl"
        stats = clip_split(source, target, args.window)
        stats["source_sha256"] = sha256(source)
        stats["output_sha256"] = sha256(target)
        manifest["splits"][split] = stats

    # Timelines are window-independent (full-dialogue roll-forward); copy so
    # STAGE0_DIR can point at this directory alone.
    for name in ("dialogue_timelines_train.json", "dialogue_timelines_validation.json",
                 "dialogue_timelines_test.json", "stage0_stats.json"):
        source = stage0 / name
        if source.exists():
            shutil.copy2(source, output / name)
            manifest["copies"][name] = sha256(source)

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
