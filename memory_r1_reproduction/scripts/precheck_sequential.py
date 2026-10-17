"""Paired, causal-prefix controls on TRAIN tuples; no fitted threshold by default."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_manager_grpo import TextModel, read_rows, normalize_tuples
from evaluation.metrics import f1_score, exact_match
from memfactory.locomo_split import SPLIT_DIALOGUE_IDS
from memfactory.memory_runtime import retrieve, retrieve_per_speaker, apply_decisions, parse_manager_output
from memfactory.modules.memory_updater import build_manager_input
from memfactory.oracle_reward import (evaluate_controls, safe_answer, teacher_fact_memories,
                                      ContextBudgetError, gold_text, append_jsonl)
from memfactory.run_metadata import write_run_metadata
from memfactory.training_state import preserve_rng


def precheck_item(item, manager, answer, args, reward_fn):
    with preserve_rng():
        controls = evaluate_controls(answer, item, args, reward_fn)
        bank, invalid = [], 0
        for turn in item["dialogue_turns"]:
            facts = [dict(f, speaker=f.get("speaker", turn.get("speaker", "")),
                          timestamp=f.get("timestamp", turn.get("timestamp", "")))
                     for f in turn["facts"] if isinstance(f, dict) and str(f.get("text", "")).strip()]
            if not facts:
                continue
            visible = retrieve(" ".join(f["text"] for f in facts), bank, args.manager_top_k)
            raw, _ = manager.generate(build_manager_input(visible, facts), args.max_new_tokens, 0.0)
            try:
                apply_decisions(bank, parse_manager_output(raw), item["dialogue_id"], turn,
                                visible_memory_ids={entry["id"] for entry in visible})
            except (ValueError, KeyError, TypeError):
                invalid += 1
        records = []
        for question, control in zip(item["linked_questions"], controls):
            conditions = {"empty_noop": control["noop"], "teacher_all_facts": control["teacher"]}
            inputs = {"frozen_manager": retrieve_per_speaker(question["question"], bank,
                                                            item["participants"], args.answer_top_k)}
            if args.include_gold_support:
                evidence = set(question.get("evidence", []))
                inputs["gold_support_facts"] = teacher_fact_memories(
                    [turn for turn in item["dialogue_turns"] if turn["dia_id"] in evidence])
            if args.include_raw:
                inputs["raw_prefix"] = [{"speaker": t["speaker"], "timestamp": t.get("timestamp", ""),
                                          "text": t["text"]} for t in item["dialogue_turns"]]
            for name, memories in inputs.items():
                try:
                    result = safe_answer(answer, question["question"], memories, item["participants"],
                                         args.answer_max_new_tokens, args.context_overflow, args.compression_tokens)
                    result["score"] = reward_fn(result["prediction"], gold_text(question["answer"]))
                except ContextBudgetError as exc:
                    result = {"status": "context_error", "error": str(exc), "score": None}
                conditions[name] = result
            records.append({"dialogue_id": item["dialogue_id"], "turn_index": item["turn_index"],
                            "question_id": control["question_id"], "question": question["question"],
                            "gold": gold_text(question["answer"]), "eligible": control["eligible"],
                            "invalid_manager_actions": invalid, "conditions": conditions})
        return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--manager-model", required=True)
    parser.add_argument("--manager-adapter")
    parser.add_argument("--answer-model", required=True)
    # Manager precheck follows Memory-R1's first stage: the frozen reader is
    # the original base model. An adapter is optional for an explicitly
    # separate ablation, but must not be required for the paper-faithful path.
    parser.add_argument("--answer-adapter")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reward", choices=["f1", "em"], default="f1",
                        help="Control score; defaults to baseline-lora F1. Match the Manager training reward.")
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--answer-max-new-tokens", type=int, default=2048)
    parser.add_argument("--manager-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k", type=int, default=30)
    parser.add_argument("--context-overflow", choices=["compress", "error"], default="compress")
    parser.add_argument("--compression-tokens", type=int, default=256)
    parser.add_argument("--control-threshold", type=float)
    parser.add_argument("--include-gold-support", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--limit-tuples", type=int, help="Smoke test only; omit for calibration")
    args = parser.parse_args()
    args.control_gate = "off" if args.control_threshold is None else "threshold"
    if args.control_threshold is not None and not 0 <= args.control_threshold <= 1:
        parser.error("control-threshold must be in [0,1]")
    if args.limit_tuples is not None and args.limit_tuples < 1:
        parser.error("limit-tuples must be positive")
    rows = normalize_tuples(read_rows(args.data_path))
    if not rows or any(r["split"] != "train" or r["dialogue_id"] not in SPLIT_DIALOGUE_IDS["train"] for r in rows):
        raise ValueError("Calibrate only on the fixed training dialogues")
    if args.limit_tuples:
        rows = rows[:args.limit_tuples]
    output = Path(args.output_dir)
    if (output / "run_manifest.json").exists():
        raise ValueError("Use a new precheck output directory")
    write_run_metadata(args)
    manager = TextModel(args.manager_model, args.device, False, args.manager_adapter, args.max_prompt_tokens)
    answer = TextModel(args.answer_model, args.device, False, args.answer_adapter, args.max_prompt_tokens)
    reward_fn = f1_score if args.reward == "f1" else exact_match
    all_records = []
    for index, row in enumerate(rows):
        for record in precheck_item(row, manager, answer, args, reward_fn):
            record["tuple_index"] = index
            append_jsonl(output / "paired.jsonl", record)
            all_records.append(record)
    scores = [r["conditions"]["teacher_all_facts"]["score"] for r in all_records]
    valid = [score for score in scores if score is not None]
    summary = {"qa": len(scores), "valid_control": len(valid), "context_errors": len(scores)-len(valid),
               "teacher_score_distribution": sorted(valid),
               "threshold_sweep": {str(t): sum(s >= t for s in valid) for t in [0, .25, .5, .75, 1]},
               "note": "Choose threshold on TRAIN only; keep identical across ablations. Full test is unfiltered."}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
