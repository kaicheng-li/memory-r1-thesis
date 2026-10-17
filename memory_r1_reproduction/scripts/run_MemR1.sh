#!/bin/bash
set -euo pipefail

# Paper-faithful two-stage Memory-R1 pipeline:
#   1. train Manager while the reward reader is the frozen base model;
#   2. replay the selected Manager to build Answer tuples;
#   3. train this run's Answer Agent on those tuples.
#
# Do not pass a trained Answer adapter to train_manager_grpo.py. That would
# make Manager training depend on a later stage (or on another experiment).

cd "$(dirname "$0")/.."

DATA_PATH="${DATA_PATH:-./data/review/manager_tuples_train_window24.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/models/${MODEL_NAME}}"
MANAGER_OUTPUT_DIR="${MANAGER_OUTPUT_DIR:-${OUTPUT_DIR:-./output/ev_manager_${MODEL_NAME}}}"
ANSWER_DATA_PATH="${ANSWER_DATA_PATH:-${MANAGER_OUTPUT_DIR}/answer_training_data.jsonl}"
ANSWER_OUTPUT_DIR="${ANSWER_OUTPUT_DIR:-./output/ev_answer_${MODEL_NAME}}"
VALIDATION_TIMELINES="${VALIDATION_TIMELINES:-./data/stage0/dialogue_timelines_validation.json}"

LR="${LR:-1e-6}"
BETA="${BETA:-0.01}"
MAX_GEN_LEN="${MAX_GEN_LEN:-2048}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
NUM_GENS="${NUM_GENS:-8}"
EPOCHS="${EPOCHS:-5}"
# Keep the agreed baseline-lora F1 reward across precheck and training.
QA_REWARD="${QA_REWARD:-f1}"
ANSWER_TOP_K_PER_SPEAKER="${ANSWER_TOP_K_PER_SPEAKER:-30}"
MANAGER_TOP_K="${MANAGER_TOP_K:-5}"
CONTROL_GATE="${CONTROL_GATE:-threshold}"
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-noop}"
EVIDENCE_WEIGHT="${EVIDENCE_WEIGHT:-0.5}"
SHAPING_MODE="${SHAPING_MODE:-dense}"
TERMINAL_POTENTIAL="${TERMINAL_POTENTIAL:-observed}"
GAMMA="${GAMMA:-1.0}"

if [ ! -f "$DATA_PATH" ]; then
    echo "Missing Manager training tuples: $DATA_PATH" >&2
    exit 1
fi
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "Missing base model config: $MODEL_PATH/config.json" >&2
    exit 1
fi

export MEMORY_RETRIEVAL_BACKEND="${MEMORY_RETRIEVAL_BACKEND:-minilm}"
export MEMORY_EMBEDDING_MODEL="${MEMORY_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

manager_extra=()
if [ -n "${MANAGER_ADAPTER_PATH:-}" ]; then
    manager_extra+=(--manager-adapter "$MANAGER_ADAPTER_PATH")
fi
if [ -n "${RESUME:-}" ]; then
    manager_extra+=(--resume "$RESUME")
fi
if [ "$CONTROL_GATE" = "threshold" ]; then
    : "${CONTROL_THRESHOLD:?Run precheck_sequential.py on the train split and set CONTROL_THRESHOLD}"
    manager_extra+=(--control-threshold "$CONTROL_THRESHOLD")
fi

echo "[1/3] Training Manager with frozen base Answer reader"
python3 scripts/train_manager_grpo.py \
    --manager-model "$MODEL_PATH" \
    --answer-model "$MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --validation-timelines "$VALIDATION_TIMELINES" \
    --output-dir "$MANAGER_OUTPUT_DIR" \
    --learning-rate "$LR" \
    --beta "$BETA" \
    --max-new-tokens "$MAX_GEN_LEN" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --num-generations "$NUM_GENS" \
    --manager-top-k "$MANAGER_TOP_K" \
    --answer-top-k "$ANSWER_TOP_K_PER_SPEAKER" \
    --device cuda \
    --epochs "$EPOCHS" \
    --seed "${SEED:-42}" \
    --reward "$QA_REWARD" \
    --answer-temperature 0.0 \
    --evidence-weight "$EVIDENCE_WEIGHT" \
    --terminal-potential "$TERMINAL_POTENTIAL" \
    --gamma "$GAMMA" \
    --advantage-estimator "$ADVANTAGE_ESTIMATOR" \
    --adv-scale "${ADV_SCALE:-1}" \
    --adv-clip "${ADV_CLIP:-5}" \
    --control-gate "$CONTROL_GATE" \
    --shaping-mode "$SHAPING_MODE" \
    --context-overflow "${CONTEXT_OVERFLOW:-compress}" \
    --compression-tokens "${COMPRESSION_TOKENS:-256}" \
    --group-log "${GROUP_LOG:-$MANAGER_OUTPUT_DIR/groups.jsonl}" \
    --checkpoint-every "${CHECKPOINT_EVERY:-50}" \
    "${manager_extra[@]}"

# Select the validation-best Manager when available. If no validation update
# improved the initial checkpoint, use the final epoch checkpoint.
if [ -n "${ANSWER_MANAGER_ADAPTER_PATH:-}" ]; then
    SELECTED_MANAGER_ADAPTER="$ANSWER_MANAGER_ADAPTER_PATH"
elif [ -f "$MANAGER_OUTPUT_DIR/best_checkpoint.json" ]; then
    SELECTED_MANAGER_ADAPTER="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); print(d.get("adapter") or "")' "$MANAGER_OUTPUT_DIR/best_checkpoint.json")"
else
    SELECTED_MANAGER_ADAPTER=""
fi
if [ -z "$SELECTED_MANAGER_ADAPTER" ] || [ ! -f "$SELECTED_MANAGER_ADAPTER/adapter_config.json" ]; then
    SELECTED_MANAGER_ADAPTER="$MANAGER_OUTPUT_DIR/epoch_${EPOCHS}"
fi
if [ ! -f "$SELECTED_MANAGER_ADAPTER/adapter_config.json" ]; then
    echo "Could not locate selected Manager adapter: $SELECTED_MANAGER_ADAPTER" >&2
    exit 1
fi

echo "[2/3] Building Answer tuples with Manager adapter $SELECTED_MANAGER_ADAPTER"
if [ -z "${REUSE_ANSWER_DATA:-}" ] || [ ! -f "$ANSWER_DATA_PATH" ]; then
    python3 scripts/build_answer_training_data.py \
        --manager-tuples "$DATA_PATH" \
        --split train \
        --output "$ANSWER_DATA_PATH" \
        --manager-model "$MODEL_PATH" \
        --manager-adapter "$SELECTED_MANAGER_ADAPTER" \
        --manager-device cuda \
        --manager-top-k "$MANAGER_TOP_K" \
        --per-speaker-top-k "$ANSWER_TOP_K_PER_SPEAKER" \
        --max-new-tokens "$MAX_GEN_LEN" \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS"
else
    echo "Reusing existing Answer tuples: $ANSWER_DATA_PATH"
fi

echo "[3/3] Training this run's Answer Agent"
if [ -z "${REUSE_ANSWER_ADAPTER:-}" ] || [ ! -f "$ANSWER_OUTPUT_DIR/epoch_${EPOCHS}/adapter_config.json" ]; then
    python3 scripts/train_answer_grpo.py \
        --data-path "$ANSWER_DATA_PATH" \
        --model-path "$MODEL_PATH" \
        --output-dir "$ANSWER_OUTPUT_DIR" \
        --learning-rate "$LR" \
        --beta "$BETA" \
        --max-new-tokens "$MAX_GEN_LEN" \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --num-generations "$NUM_GENS" \
        --epochs "$EPOCHS" \
        --seed "${ANSWER_SEED:-42}" \
        --device cuda
fi

echo "Manager adapter: $SELECTED_MANAGER_ADAPTER"
echo "Answer adapter: $ANSWER_OUTPUT_DIR/epoch_${EPOCHS}"
