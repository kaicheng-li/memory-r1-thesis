#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
if [ -n "${EVIDENCE_WEIGHT+x}" ] || [ -n "${CONTROL_THRESHOLD+x}" ] || [ -n "${ADVANTAGE_ESTIMATOR+x}" ]; then
  echo "Legacy D3 flags are not accepted: select RUN=B or RUN=C; this protocol fixes GRPO and gate off." >&2
  exit 2
fi
# B/C share the entire protocol. RUN changes ONLY eta and output locations.
RUN="${RUN:-C}"
case "$RUN" in
  B) ETA=0 ;;
  C) ETA=0.5 ;;
  *) echo "RUN must be B or C" >&2; exit 2 ;;
esac
: "${MODEL_PATH:?Set MODEL_PATH to the same base Qwen checkpoint used by A}"
RUN_ROOT="${RUN_ROOT:-./output/abc_fullprefix_${RUN}}"
DATA_PATH="${DATA_PATH:-./data/review/manager_tuples_train_fullprefix.jsonl}"
TRAIN_TIMELINES="${TRAIN_TIMELINES:-./data/stage0/dialogue_timelines_train.json}"
VALIDATION_TIMELINES="${VALIDATION_TIMELINES:-./data/stage0/dialogue_timelines_validation.json}"
MANAGER_OUTPUT_DIR="$RUN_ROOT/manager"
ANSWER_OUTPUT_DIR="$RUN_ROOT/answer"
ANSWER_DATA_PATH="$RUN_ROOT/answer_train.jsonl"
ANSWER_VALIDATION_PATH="$RUN_ROOT/answer_validation.jsonl"
# Use the same overrides for A/B/C. Manager currently runs one GPU; do not
# launch independent unsynchronized Manager copies with torchrun/accelerate.
NUM_GENS="${NUM_GENS:-4}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-5}"
BETA="${BETA:-0.01}"
SEED="${SEED:-42}"
export MEMORY_RETRIEVAL_BACKEND=minilm
export MEMORY_EMBEDDING_MODEL="${MEMORY_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
if [ ! -f "$DATA_PATH" ]; then
  python3 scripts/build_sequential_from_stage0.py --timelines "$TRAIN_TIMELINES" \
    --states ./data/stage0/manager_states_train.jsonl --split train --history-window -1 --output "$DATA_PATH"
fi
python3 scripts/check_abc_data.py --data-path "$DATA_PATH" --states ./data/stage0/manager_states_train.jsonl
extra=()
if [ -n "${RESUME:-}" ]; then extra+=(--resume "$RESUME"); fi
if [ "${CONTROL_DIAGNOSTICS:-0}" = 1 ]; then extra+=(--control-diagnostics); fi
python3 scripts/train_manager_grpo.py \
  --manager-model "$MODEL_PATH" --answer-model "$MODEL_PATH" \
  --data-path "$DATA_PATH" --validation-timelines "$VALIDATION_TIMELINES" \
  --output-dir "$MANAGER_OUTPUT_DIR" --device cuda \
  --reward f1 --advantage-estimator grpo --control-gate off \
  --evidence-weight "$ETA" --gamma 1 --terminal-potential observed --shaping-mode dense \
  --num-generations "$NUM_GENS" --epochs "$EPOCHS" --seed "$SEED" \
  --learning-rate "$LR" --warmup-steps 10 --weight-decay 0.01 --beta "$BETA" \
  --variance-epsilon 1e-12 --manager-top-k 5 --answer-top-k 30 \
  --max-prompt-tokens 4096 --max-new-tokens 2048 --answer-max-new-tokens 2048 \
  --context-overflow truncate --checkpoint-every "${CHECKPOINT_EVERY:-50}" "${extra[@]}"
if [ -f "$MANAGER_OUTPUT_DIR/stopped_early.json" ]; then
  echo "Manager acceptance gate failed; inspect stopped_early.json before continuing" >&2
  exit 1
fi
# Empty adapter means the base model genuinely won validation. Never replace
# that selection with a worse last-epoch adapter.
SELECTED_MANAGER_ADAPTER=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter"] or "")' "$MANAGER_OUTPUT_DIR/best_checkpoint.json")
manager_extra=()
if [ -n "$SELECTED_MANAGER_ADAPTER" ]; then manager_extra+=(--manager-adapter "$SELECTED_MANAGER_ADAPTER"); fi
for split in train validation; do
  if [ "$split" = train ]; then timeline="$TRAIN_TIMELINES"; output="$ANSWER_DATA_PATH"
  else timeline="$VALIDATION_TIMELINES"; output="$ANSWER_VALIDATION_PATH"; fi
  python3 scripts/build_answer_training_data.py --timeline "$timeline" --split "$split" \
    --output "$output" --manager-model "$MODEL_PATH" --manager-device cuda \
    --manager-top-k 5 --per-speaker-top-k 30 --max-prompt-tokens 4096 --max-new-tokens 2048 \
    "${manager_extra[@]}"
done
python3 scripts/train_answer_grpo.py --data-path "$ANSWER_DATA_PATH" \
  --validation-data-path "$ANSWER_VALIDATION_PATH" --split all \
  --model-path "$MODEL_PATH" --output-dir "$ANSWER_OUTPUT_DIR" \
  --num-generations "$NUM_GENS" --epochs "$EPOCHS" --seed "$SEED" \
  --learning-rate "$LR" --warmup-steps 10 --weight-decay 0.01 --beta "$BETA" \
  --variance-epsilon 1e-12 --max-prompt-tokens 4096 --max-new-tokens 2048
printf 'Manager selection: %s/best_checkpoint.json\nAnswer selection: %s/best_checkpoint.json\n' "$MANAGER_OUTPUT_DIR" "$ANSWER_OUTPUT_DIR"
