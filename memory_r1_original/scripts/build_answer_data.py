from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_r1.agents import FactExtractor
from memory_r1.datasets import answer_sample_to_row, build_answer_samples
from memory_r1.utils import dump_jsonl, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Answer Agent training tuples.")
    parser.add_argument("--input", required=True, help="Dialogue JSON file.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--per_speaker_top_k", type=int, default=30)
    args = parser.parse_args()

    dialogues = load_json(args.input)
    extractor = FactExtractor(generator=None)
    samples = build_answer_samples(dialogues, extractor, per_speaker_top_k=args.per_speaker_top_k)
    dump_jsonl(args.output, (answer_sample_to_row(sample) for sample in samples))
    print(f"wrote {len(samples)} answer samples to {args.output}")


if __name__ == "__main__":
    main()
