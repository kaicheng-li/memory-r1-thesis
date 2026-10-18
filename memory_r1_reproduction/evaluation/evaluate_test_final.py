"""Stage 4: final LoCoMo test evaluation (1307 QA, local fixed split).

Semantics are identical to training-time roll-forward validation:
  cached teacher facts -> best Manager greedy roll-forward builds each test
  dialogue's bank -> best Answer answers every question greedily ->
  F1 / BLEU-1 / (optional LLM-as-a-Judge), overall and per category.

Run per-dialogue (parallel across GPUs), then merge:

  # one process per GPU, dialogues split across them
  CUDA_VISIBLE_DEVICES=0 python3 evaluation/evaluate_test_final.py \
      --model-path <BASE_MODEL> --dialogues conv-41,conv-42 &
  ...

  # after all 8 dialogues finished:
  python3 evaluation/evaluate_test_final.py --merge
  # with judge:
  python3 evaluation/evaluate_test_final.py --merge \
      --judge-base-url http://... --judge-model gpt-4o-mini --judge-api-key KEY

Each dialogue writes <output-dir>/<dialogue_id>.json and is skipped on rerun,
so a crash never repeats finished dialogues.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import LOCOMO_CATEGORY_NAMES, normalize_text, exact_match, f1_score

JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.

You will be given the question, the gold (ground-truth) answer, and a generated answer, which you will score as CORRECT or WRONG.

The generated answer might be longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time-related questions, the gold answer will be a specific date, month, or year. The generated answer might include relative references (e.g., "last Tuesday"), but you should be generous - if it refers to the same time period as the gold answer, mark it CORRECT, even if the format differs (e.g., "May 7th" vs. "7 May").

Question: {question}
Gold answer: {gold}
Generated answer: {prediction}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response.
"""


def resolve_adapter(explicit: str | None, output_dir: str) -> str:
    if explicit:
        return explicit
    best = Path(output_dir) / "best_checkpoint.json"
    if best.is_file():
        record = json.loads(best.read_text(encoding="utf-8"))
        return record["adapter"] if "adapter" in record else record["path"]
    epochs = sorted(
        Path(output_dir).glob("epoch_*"),
        key=lambda p: int(re.sub(r"\D", "", p.name) or 0),
    )
    if not epochs:
        raise SystemExit(f"No adapter found under {output_dir}")
    return str(epochs[-1])


def evaluate_dialogues(args: argparse.Namespace) -> None:
    import torch

    from scripts.train_manager_grpo import TextModel
    from memfactory.sequential_validation import evaluate_roll_forward
    from memfactory.oracle_reward import safe_answer
    from memfactory.retrieval import retrieval_config
    from transformers import set_seed
    set_seed(args.seed)
    args.answer_top_k = args.answer_top_k_per_speaker

    manager_adapter = resolve_adapter(args.manager_adapter, args.manager_output_dir)
    answer_adapter = resolve_adapter(args.answer_adapter, args.answer_output_dir)
    print(f"manager adapter: {manager_adapter}", flush=True)
    print(f"answer adapter:  {answer_adapter}", flush=True)

    timelines = json.loads(Path(args.timelines).read_text(encoding="utf-8"))
    wanted = [d.strip() for d in args.dialogues.split(",") if d.strip()] \
        if args.dialogues else [t["dialogue_id"] for t in timelines]
    timelines = {t["dialogue_id"]: t for t in timelines}
    missing = [d for d in wanted if d not in timelines]
    if missing:
        raise SystemExit(f"dialogues not in timeline file: {missing}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manager = TextModel(args.model_path, args.device, False, manager_adapter, args.max_prompt_tokens)
    answer = TextModel(args.model_path, args.device, False, answer_adapter, args.max_prompt_tokens)

    def manager_generate(prompt):
        return manager.generate(prompt, args.max_new_tokens, 0.0)[0]

    def answer_generate(question, memories, participants):
        return safe_answer(answer, question, memories, participants, args.answer_max_new_tokens,
                           args.context_overflow, args.compression_tokens)

    for dialogue_id in wanted:
        out_path = output_dir / f"{dialogue_id}.json"
        if out_path.is_file() and not args.force:
            print(f"[{dialogue_id}] already done, skipping", flush=True)
            continue
        timeline = timelines[dialogue_id]
        n_turns = len(timeline["turns"])
        n_qa = len(timeline["questions"])
        print(f"[{dialogue_id}] roll-forward: {n_turns} turns, {n_qa} QA ...", flush=True)
        started = time.time()
        result = evaluate_roll_forward(
            [timeline], manager_generate, answer_generate, args, normalize_text, exact_match, f1_score
        )
        result["meta"] = {
            "dialogue_id": dialogue_id,
            "manager_adapter": manager_adapter,
            "answer_adapter": answer_adapter,
            "model_path": args.model_path,
            "decoding": "greedy",
            "retrieval": retrieval_config(),
            "context_overflow": args.context_overflow,
            "compression_tokens": args.compression_tokens,
            "elapsed_s": time.time() - started,
        }
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        s = result["summary"]
        print(
            f"[{dialogue_id}] done in {result['meta']['elapsed_s']/60:.1f} min | "
            f"QA={s['qa']} F1={s['f1']:.4f} B1={s['bleu1']:.4f} EM={s['em']:.4f} "
            f"invalid_turns={s['invalid_turns']}",
            flush=True,
        )


def judge_records(records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    from openai import OpenAI

    client = OpenAI(base_url=args.judge_base_url,
                    api_key=args.judge_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
    for index, record in enumerate(records, start=1):
        if "judge" in record:
            continue
        prompt = JUDGE_PROMPT.format(
            question=record["question"], gold=record["gold"],
            prediction=record["prediction"] or "(empty)",
        )
        response = client.chat.completions.create(
            model=args.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=256,
        )
        text = (response.choices[0].message.content or "").upper()
        labels = set(re.findall(r"\b(CORRECT|WRONG)\b", text))
        record["judge"] = 1.0 if labels == {"CORRECT"} else 0.0 if labels == {"WRONG"} else None
        record["judge_raw"] = response.choices[0].message.content
        if index % 50 == 0:
            print(f"judged {index}/{len(records)}", flush=True)


def merge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    files = sorted(output_dir.glob("conv-*.json"))
    if not files:
        raise SystemExit(f"no per-dialogue results in {output_dir}")
    records: list[dict[str, Any]] = []
    invalid_turns = 0
    total_turns = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload["records"])
        invalid_turns += payload["summary"]["invalid_turns"]
        total_turns += sum(d["turns"] for d in payload["dialogues"])
    print(f"merged {len(files)} dialogues, {len(records)} QA records")

    if args.judge_base_url:
        judge_records(records, args)
        judged_path = output_dir / "records_judged.jsonl"
        judged_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8",
        )

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        judged = [r for r in rows if r.get("judge") is not None]
        return {
            "n": n,
            "f1": 100 * sum(r["f1"] for r in rows) / n if n else 0.0,
            "token_f1_multiset": 100 * sum(r["token_f1_multiset"] for r in rows) / n if n else 0.0,
            "bleu1": 100 * sum(r["bleu1"] for r in rows) / n if n else 0.0,
            "em": 100 * sum(r["em"] for r in rows) / n if n else 0.0,
            "judge": (100 * sum(r["judge"] for r in judged) / len(judged))
                     if judged else None,
            "judge_n": len(judged),
        }

    summary: dict[str, Any] = {
        "scope": "locomo_test_1307",
        "dialogues": len(files),
        "invalid_turns": invalid_turns,
        "total_turns": total_turns,
        "overall": block(records),
        "by_category": {},
    }
    for cat, name in LOCOMO_CATEGORY_NAMES.items():
        rows = [r for r in records if int(r.get("category", 0)) == cat]
        if rows:
            summary["by_category"][name] = block(rows)

    (output_dir / "final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    o = summary["overall"]
    print(f"\n=== LoCoMo test (local fixed split, percent) ===")
    print(f"QA={o['n']}  F1={o['f1']:.2f}  B1={o['bleu1']:.2f}  "
          f"EM={o['em']:.2f}  J={o['judge'] if o['judge'] is not None else '-'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timelines", default="data/stage0/dialogue_timelines_test.json")
    parser.add_argument("--output-dir", default="output/final_test_eval")
    parser.add_argument("--model-path", help="Base model (required unless --merge)")
    parser.add_argument("--manager-adapter")
    parser.add_argument("--answer-adapter")
    parser.add_argument("--manager-output-dir",
                        default="output/mem_r1_manager_Qwen2.5-7B-Instruct")
    parser.add_argument("--answer-output-dir",
                        default="output/mem_r1_answer_Qwen2.5-7B-Instruct")
    parser.add_argument("--dialogues", help="Comma list, e.g. conv-41,conv-42; default all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k-per-speaker", type=int, default=30)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--answer-max-new-tokens", type=int, default=2048)
    parser.add_argument("--context-overflow", choices=["truncate", "compress", "error"], default="truncate")
    parser.add_argument("--compression-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--judge-api-key")
    args = parser.parse_args()
    if args.merge:
        merge(args)
    else:
        if not args.model_path:
            parser.error("--model-path is required for evaluation")
        evaluate_dialogues(args)


if __name__ == "__main__":
    main()
