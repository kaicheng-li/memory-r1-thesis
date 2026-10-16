"""Reject automatic reuse of Answer adapters built with stale data/prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memfactory.run_metadata import PROMPT_CONTRACT, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--manager-tuples", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-prompt-tokens", required=True, type=int)
    args = parser.parse_args()
    try:
        manifest = json.loads((Path(args.output_dir) / "run_manifest.json").read_text(encoding="utf-8"))
        data_path = Path(args.data_path)
        with data_path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        config = manifest["config"]
        source_hash = file_sha256(args.manager_tuples)
        valid = (
            manifest["prompt_contract"] == PROMPT_CONTRACT
            and manifest["data_sha256"] == file_sha256(data_path)
            and config["model_path"] == args.model_path
            and config["max_prompt_tokens"] == args.max_prompt_tokens
            and rows
            and all(row.get("metadata", {}).get("source") == "manager_tuple_window_replay_v1"
                    and row.get("metadata", {}).get("manager_tuples_sha256") == source_hash
                    and isinstance(row.get("participants"), list) for row in rows)
        )
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        print("Answer artifact has missing/stale data or prompt metadata; rebuild or explicitly select an audited adapter.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
