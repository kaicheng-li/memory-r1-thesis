"""Stage 0.5: paired counterfactual precheck (go/no-go gate before Manager GRPO).

For every Stage-0 training state, answer the SAME question twice with the
frozen base Answer model under identical retrieval and greedy decoding:

    noop   : old_bank only                      (Manager does nothing)
    oracle : old_bank + current-turn facts      (Manager performs perfect ADDs)

Continuous token F1 is the planned RL reward. The report therefore records its
mean and paired deltas directly. A separate layered binary matcher gives an
interpretable count of QAs that flip from incorrect to correct.

Gate (frozen in review): layered_match_positive_delta_count >= 15 (10% of
150) to start training. This gate is deliberately not called EM or F1: it is
the auxiliary layered matcher. Strict EM remains a paper-facing diagnostic.

Single GPU, ~2 x 150 greedy generations (~20 min on one 48GB card).

Usage:
    # Pure-CPU rescore of an earlier precheck (preferred when predictions exist):
    python3 scripts/precheck_counterfactual.py \
        --rescore-input data/stage0/precheck_train.json \
        --output data/stage0/precheck_train_f1.json

    # Generate predictions when no earlier precheck exists:
    python3 scripts/precheck_counterfactual.py \
        --states data/stage0/manager_states_train.jsonl \
        --model-path /home/models/Qwen2.5-7B-Instruct \
        --output data/stage0/precheck_train.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.matching import match_locomo_v2


def oracle_bank(state: dict[str, Any]) -> list[dict[str, Any]]:
    """old_bank plus current-turn facts appended as environment-ADDed entries."""
    bank = list(state["old_bank"])
    dialogue_id = state["dialogue_id"]
    offset = len(bank)
    for index, fact in enumerate(state["current_facts"]):
        bank.append({
            "id": f"{dialogue_id}:precheck:m{offset + index}",
            "text": str(fact.get("text", "")).strip(),
            "speaker": str(fact.get("speaker", state["target_speaker"])),
            "timestamp": str(fact.get("timestamp", state["target_timestamp"])),
            "source_turn": state["target_turn"],
        })
    return bank


def answer_once(model: Any,
                state: dict[str, Any],
                bank: list[dict[str, Any]],
                args: argparse.Namespace) -> tuple[str, list[str]]:
    from evaluation.audit_checkpoints import extract_answer
    from memfactory.modules.memory_retriever import build_answer_input
    from scripts.construct_memory_bank import top_k_per_speaker

    retrieved = top_k_per_speaker(
        state["question"], bank, state["participants"], args.top_k_per_speaker
    )
    memories_by_speaker = {
        participant: [m for m in retrieved if m["speaker"] == participant]
        for participant in state["participants"]
    }
    prompt = build_answer_input(state["question"], memories_by_speaker)
    raw, _tokens, _hit = model.generate(
        prompt, args.max_prompt_tokens, args.max_new_tokens,
        temperature=0.0, seed=args.seed,
    )
    return extract_answer(raw), sorted(str(m.get("id", "")) for m in retrieved)


def score_predictions(record: dict[str, Any]) -> dict[str, Any]:
    """Recompute all metrics from saved prediction/gold strings."""
    noop_match = match_locomo_v2(record["noop_prediction"], record["gold"])
    oracle_match = match_locomo_v2(record["oracle_prediction"], record["gold"])
    stale_fields = {
        "noop_em", "oracle_em", "noop_match", "oracle_match",
        "noop_tier", "oracle_tier", "noop_em_strict", "oracle_em_strict",
        "noop_f1", "oracle_f1",
    }
    scored = {key: value for key, value in record.items() if key not in stale_fields}
    scored.update({
        "noop_match": noop_match["reward"],
        "oracle_match": oracle_match["reward"],
        "noop_tier": noop_match["tier"],
        "oracle_tier": oracle_match["tier"],
        "noop_em_strict": noop_match["em_strict"],
        "oracle_em_strict": oracle_match["em_strict"],
        "noop_f1": noop_match["f1"],
        "oracle_f1": oracle_match["f1"],
    })
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default="data/stage0/manager_states_train.jsonl")
    parser.add_argument("--model-path", help="Required unless --rescore-input is used")
    parser.add_argument("--adapter", help="Optional Answer adapter (default: base model)")
    parser.add_argument(
        "--rescore-input",
        help="Existing precheck JSON whose saved predictions will be rescored on CPU",
    )
    parser.add_argument("--output", default="data/stage0/precheck_train.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k-per-speaker", type=int, default=30)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gate",
        type=int,
        default=15,
        help="Minimum layered-match 0->1 transitions to pass (default: 15)",
    )
    args = parser.parse_args()

    source_summary: dict[str, Any] = {}
    if args.rescore_input:
        payload = json.loads(Path(args.rescore_input).read_text(encoding="utf-8"))
        source_records = payload.get("records", payload) if isinstance(payload, dict) else payload
        source_summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        if not isinstance(source_records, list) or not source_records:
            parser.error("--rescore-input must contain a non-empty records list")
        records = [score_predictions(record) for record in source_records]
        print(f"rescored {len(records)} saved QA pairs from {args.rescore_input}", flush=True)
    else:
        if not args.model_path:
            parser.error("--model-path is required unless --rescore-input is used")
        from evaluation.audit_checkpoints import AuditModel

        states = [json.loads(line) for line in
                  Path(args.states).read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        model = AuditModel(args.model_path, args.adapter, args.device)
        records = []
        started = time.time()
        for index, state in enumerate(states, start=1):
            noop_pred, noop_ids = answer_once(model, state, state["old_bank"], args)
            oracle_pred, oracle_ids = answer_once(model, state, oracle_bank(state), args)
            record = score_predictions({
                "dialogue_id": state["dialogue_id"],
                "question_id": state["question_id"],
                "category": state["category"],
                "target_turn": state["target_turn"],
                "gold": state["answer"],
                "noop_prediction": noop_pred,
                "oracle_prediction": oracle_pred,
                "retrieval_changed": noop_ids != oracle_ids,
            })
            records.append(record)
            elapsed = time.time() - started
            print(f"[{index}/{len(states)}] qid={record['question_id']} "
                  f"noop_f1={record['noop_f1']:.3f} "
                  f"oracle_f1={record['oracle_f1']:.3f} "
                  f"match={record['noop_match']:.0f}->{record['oracle_match']:.0f} "
                  f"({elapsed / index:.1f}s/QA)", flush=True)

    n = len(records)
    match_positive = [
        r for r in records if r["noop_match"] == 0 and r["oracle_match"] == 1
    ]
    match_negative = [
        r for r in records if r["noop_match"] == 1 and r["oracle_match"] == 0
    ]
    f1_deltas = [r["oracle_f1"] - r["noop_f1"] for r in records]
    summary = {
        "states": n,
        "training_reward_rule": "continuous_token_f1_unique_overlap",
        "diagnostic_match_rule": "match_locomo_v2",
        "noop_match_rate": sum(r["noop_match"] for r in records) / n,
        "oracle_match_rate": sum(r["oracle_match"] for r in records) / n,
        "noop_em_strict": sum(r["noop_em_strict"] for r in records) / n,
        "oracle_em_strict": sum(r["oracle_em_strict"] for r in records) / n,
        "noop_f1": sum(r["noop_f1"] for r in records) / n,
        "oracle_f1": sum(r["oracle_f1"] for r in records) / n,
        "f1_mean_delta": sum(f1_deltas) / n,
        "f1_positive_delta_count": sum(delta > 1e-12 for delta in f1_deltas),
        "f1_negative_delta_count": sum(delta < -1e-12 for delta in f1_deltas),
        "f1_delta_gt_0_2_count": sum(delta > 0.2 for delta in f1_deltas),
        "layered_match_positive_delta_count": len(match_positive),
        "layered_match_negative_delta_count": len(match_negative),
        "layered_match_positive_delta_qids": [
            r["question_id"] for r in match_positive
        ],
        "layered_match_negative_delta_qids": [
            r["question_id"] for r in match_negative
        ],
        "retrieval_changed_count": sum(r["retrieval_changed"] for r in records),
        "oracle_tiers": dict(__import__("collections").Counter(
            r["oracle_tier"] for r in records)),
        "gate_rule": "layered_match_positive_delta_count >= gate_threshold",
        "gate_threshold": args.gate,
        "gate_passed": len(match_positive) >= args.gate,
        "model_path": args.model_path or source_summary.get("model_path"),
        "adapter": args.adapter if args.adapter is not None else source_summary.get("adapter"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "records": records},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output}")
    print("GATE PASSED" if summary["gate_passed"] else
          "GATE FAILED: do NOT start Manager training; fix extraction/retrieval first.")


if __name__ == "__main__":
    main()
