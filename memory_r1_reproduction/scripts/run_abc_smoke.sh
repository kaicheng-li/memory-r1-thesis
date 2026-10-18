#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?Set MODEL_PATH to the base Qwen checkpoint}"
RUN="${RUN:-B}"
case "$RUN" in B) ETA=0 ;; C) ETA=0.5 ;; *) exit 2 ;; esac
DATA_PATH="${DATA_PATH:-./data/review/manager_tuples_train_fullprefix.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/abc_smoke_${RUN}}"
export MEMORY_RETRIEVAL_BACKEND=minilm
# Smoke only: no validation, no Answer training, no claims about model quality.
# Keep real token budgets so timings remain relevant to the intended run.
python3 scripts/train_manager_grpo.py \
  --data-path "$DATA_PATH" --manager-model "$MODEL_PATH" --answer-model "$MODEL_PATH" \
  --output-dir "$OUTPUT_DIR" --device cuda --limit-tuples 2 --epochs 1 --num-generations 2 \
  --evidence-weight "$ETA" --reward f1 --advantage-estimator grpo --control-gate off \
  --learning-rate 1e-5 --warmup-steps 10 --weight-decay 0.01 --beta 0.01 \
  --gamma 1 --variance-epsilon 1e-12 --context-overflow truncate --checkpoint-every 1 \
  --max-prompt-tokens 4096 --max-new-tokens 2048 --answer-max-new-tokens 2048
