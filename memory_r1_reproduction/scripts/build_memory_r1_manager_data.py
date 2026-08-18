from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_r1.agents import FactExtractor
from memory_r1.datasets import build_manager_samples, manager_sample_to_row
from memory_r1.utils import dump_jsonl, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Memory Manager training tuples.")
    parser.add_argument("--input", required=True, help="Dialogue JSON file.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--history_window", type=int, default=50)
    args = parser.parse_args()

    dialogues = load_json(args.input)
    extractor = FactExtractor(generator=None)
    samples = build_manager_samples(dialogues, extractor, history_window=args.history_window)
    dump_jsonl(args.output, (manager_sample_to_row(sample) for sample in samples))
    print(f"wrote {len(samples)} manager samples to {args.output}")


if __name__ == "__main__":
    main()
