#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?Set MODEL_PATH to the local base Qwen checkpoint}"
RUN="${RUN:-S-A}"
case "$RUN" in
  S-A) MODE=grpo ;;
  S-B) MODE=fixed-zero ;;
  S-C) MODE=raw-reference ;;
  *) echo "RUN must be S-A, S-B or S-C (single-step protocol)" >&2; exit 2 ;;
esac
PHASE="${PHASE:-all}"
case "$PHASE" in reference|manager|all) ;; *) echo "PHASE must be reference, manager or all" >&2; exit 2 ;; esac
if [ -n "${EVIDENCE_WEIGHT+x}" ] || [ -n "${CONTROL_THRESHOLD+x}" ] || [ -n "${ADVANTAGE_ESTIMATOR+x}" ]; then
  echo "Unset legacy dense/noop/gate environment variables before this single-step run" >&2
  exit 2
fi
NUM_GPUS="${NUM_GPUS:-4}"
RUN_ROOT="${RUN_ROOT:-./output/reference_${RUN}}"
CACHE_DIR="${CACHE_DIR:-./output/raw_reference_cache}"
DATA_PATH="${DATA_PATH:-./data/stage0/manager_states_train.jsonl}"
TRAIN_TIMELINES="${TRAIN_TIMELINES:-./data/stage0/dialogue_timelines_train.json}"
VALIDATION_TIMELINES="${VALIDATION_TIMELINES:-./data/stage0/dialogue_timelines_validation.json}"
VALIDATION_STATES="${VALIDATION_STATES:-./data/stage0/manager_states_validation.jsonl}"
NUM_GENS="${NUM_GENS:-4}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-42}"
LR="${LR:-1e-5}"
BETA="${BETA:-0.01}"
mkdir -p "$RUN_ROOT" logs
LOG_FILE="${LOG_FILE:-logs/reference_${RUN}.log}"
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
export MEMORY_RETRIEVAL_BACKEND=minilm
export MEMORY_EMBEDDING_MODEL="${MEMORY_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
if [ "$NUM_GPUS" = 1 ]; then launch=(python3 -u)
elif [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  launch=(python3 -m torch.distributed.run --standalone --nproc_per_node "$NUM_GPUS")
else echo "NUM_GPUS must be a positive integer" >&2; exit 2; fi
extra=()
if [ "$PHASE" = reference ]; then extra+=(--reference-only); fi
if [ -n "${RESUME:-}" ]; then extra+=(--resume-from "$RESUME"); fi
if [ -n "${COMPARISON_LOG:-}" ]; then extra+=(--comparison-log "$COMPARISON_LOG"); fi
"${launch[@]}" scripts/train_manager_reference.py \
  --manager-model "$MODEL_PATH" --answer-model "$MODEL_PATH" \
  --data-path "$DATA_PATH" --validation-states "$VALIDATION_STATES" \
  --validation-timelines "$VALIDATION_TIMELINES" --reference-timelines "$TRAIN_TIMELINES" \
  --reference-cache-dir "$CACHE_DIR" --reference-max-prompt-tokens 30720 \
  --output-dir "$RUN_ROOT/manager" --advantage-mode "$MODE" --advantage-scale 1 \
  --acceptance-policy report-only --num-generations "$NUM_GENS" --epochs "$EPOCHS" --seed "$SEED" \
  --learning-rate "$LR" --weight-decay 0.01 --warmup-steps 10 --beta "$BETA" \
  --clip-epsilon 0.2 --grad-clip-norm 1 --variance-epsilon 1e-12 --temperature 1 \
  --manager-top-k 5 --answer-top-k-per-speaker 30 \
  --max-prompt-tokens 4096 --max-new-tokens 2048 --answer-max-new-tokens 2048 \
  --save-every-groups "${CHECKPOINT_EVERY:-10}" "${extra[@]}"
if [ "$PHASE" != all ]; then exit 0; fi
SELECTED_MANAGER=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter"] or "")' "$RUN_ROOT/manager/best_checkpoint.json")
manager_extra=()
if [ -n "$SELECTED_MANAGER" ]; then manager_extra+=(--manager-adapter "$SELECTED_MANAGER"); fi
for split in train validation; do
  if [ "$split" = train ]; then timeline="$TRAIN_TIMELINES"; else timeline="$VALIDATION_TIMELINES"; fi
  python3 -u scripts/build_answer_training_data.py --timeline "$timeline" --split "$split" \
    --output "$RUN_ROOT/answer_${split}.jsonl" --manager-model "$MODEL_PATH" --manager-device cuda \
    --manager-top-k 5 --per-speaker-top-k 30 --max-prompt-tokens 4096 --max-new-tokens 2048 "${manager_extra[@]}"
done
answer_extra=()
if [ -n "${ANSWER_RESUME:-}" ]; then answer_extra+=(--resume-from "$ANSWER_RESUME"); fi
"${launch[@]}" scripts/train_answer_grpo.py --data-path "$RUN_ROOT/answer_train.jsonl" \
  --validation-data-path "$RUN_ROOT/answer_validation.jsonl" --split all \
  --model-path "$MODEL_PATH" --output-dir "$RUN_ROOT/answer" \
  --num-generations "$NUM_GENS" --epochs "$EPOCHS" --seed "$SEED" \
  --learning-rate "$LR" --weight-decay 0.01 --warmup-steps 10 --beta "$BETA" \
  --variance-epsilon 1e-12 --max-prompt-tokens 4096 --max-new-tokens 2048 "${answer_extra[@]}"
printf 'Manager selection: %s/manager/best_checkpoint.json\nAnswer selection: %s/answer/best_checkpoint.json\n' "$RUN_ROOT" "$RUN_ROOT"
