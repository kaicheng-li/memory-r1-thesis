#!/bin/bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration for MemR1 Training
# ----------------------------------------------------------------------------

# 1. Paths
# DATA_PATH must be Algorithm 1 tuples (build_manager_training_data.py
# output): per-turn episodes with 24 preceding turns plus the current turn.
DATA_PATH="${DATA_PATH:-./data/manager_training_data_evidence.jsonl}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/models/${MODEL_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/mem_r1_lora_evidence_${MODEL_NAME}}"

# Answer Agent (paper Section 3.3) training data: Algorithm 2 tuples
# (build_answer_training_data.py replays the same cached-fact windows).
ANSWER_DATA_PATH="${ANSWER_DATA_PATH:-./data/answer_training_data_evidence.jsonl}"
ANSWER_OUTPUT_DIR="${ANSWER_OUTPUT_DIR:-./output/mem_r1_lora_evidence_answer_${MODEL_NAME}}"
ANSWER_MANAGER_PATH="${ANSWER_MANAGER_PATH:-$MODEL_PATH}"
# Optional explicit override; otherwise the last Answer Agent adapter is used.
ANSWER_ADAPTER_PATH="${ANSWER_ADAPTER_PATH:-}"

# 2. Training Hyperparameters
LR="${LR:-1e-6}"
BETA="${BETA:-0.01}"
MAX_GEN_LEN="${MAX_GEN_LEN:-2048}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
NUM_GENS="${NUM_GENS:-8}"
EPOCHS="${EPOCHS:-5}"
EVIDENCE_WEIGHT="${EVIDENCE_WEIGHT:-0.5}"
TERMINAL_POTENTIAL="${TERMINAL_POTENTIAL:-observed}"
GAMMA="${GAMMA:-1.0}"
QA_REWARD="${QA_REWARD:-em}"
ANSWER_TOP_K_PER_SPEAKER="${ANSWER_TOP_K_PER_SPEAKER:-30}"
VALIDATION_TIMELINES="${VALIDATION_TIMELINES:-./data/stage0/dialogue_timelines_validation.json}"

# ----------------------------------------------------------------------------
# 1) Train the Answer Agent (Section 3.3) on Algorithm 2 tuples.
#    Algorithm 5 requires a fixed Answer Agent as the reward oracle.
# ----------------------------------------------------------------------------

cd "$(dirname "$0")/.." # Go to project root

if [ ! -f "$DATA_PATH" ]; then
    echo "Missing Manager training tuples: $DATA_PATH" >&2
    echo "Build the train split with scripts/build_manager_training_data.py --split train first." >&2
    exit 1
fi

if [ -z "$ANSWER_ADAPTER_PATH" ]; then
    if [ -f "$ANSWER_OUTPUT_DIR/epoch_${EPOCHS}/adapter_config.json" ] && [ -z "${FORCE_ANSWER_RETRAIN:-}" ] && [ -z "${FORCE_ANSWER_DATA:-}" ] && \
       python3 scripts/check_answer_artifact.py --output-dir "$ANSWER_OUTPUT_DIR" \
           --data-path "$ANSWER_DATA_PATH" --manager-tuples "$DATA_PATH" --model-path "$MODEL_PATH" --max-prompt-tokens "$MAX_PROMPT_TOKENS"; then
        echo "Answer Agent adapter found: $ANSWER_OUTPUT_DIR/epoch_${EPOCHS}"
    else
        if [ ! -f "$ANSWER_DATA_PATH" ] || [ -n "${FORCE_ANSWER_DATA:-}" ] || \
           ! python3 -c 'import json,sys; from memfactory.run_metadata import file_sha256; rows=[json.loads(x) for x in open(sys.argv[1],encoding="utf-8") if x.strip()]; source_hash=file_sha256(sys.argv[2]); sys.exit(0 if rows and all("participants" in r and r.get("metadata",{}).get("source")=="manager_tuple_window_replay_v1" and r.get("metadata",{}).get("manager_tuples_sha256")==source_hash for r in rows) else 1)' "$ANSWER_DATA_PATH" "$DATA_PATH"; then
            echo "Building Answer Agent data from Manager tuples: $DATA_PATH ..."
            python3 scripts/build_answer_training_data.py \
                --manager-tuples "$DATA_PATH" \
                --split train \
                --output "$ANSWER_DATA_PATH" \
                --manager-model "$ANSWER_MANAGER_PATH" \
                --manager-adapter "${ANSWER_MANAGER_ADAPTER_PATH:-}" \
                --manager-device "cuda" \
                --manager-top-k 5 \
                --per-speaker-top-k "$ANSWER_TOP_K_PER_SPEAKER" \
                --max-new-tokens "$MAX_GEN_LEN" \
                --max-prompt-tokens "$MAX_PROMPT_TOKENS"
        fi
        echo "Training Answer Agent on $ANSWER_DATA_PATH ..."
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
            --device "cuda"
    fi
    ANSWER_ADAPTER_PATH="$ANSWER_OUTPUT_DIR/epoch_${EPOCHS}"
fi

# ----------------------------------------------------------------------------
# 2) Train the Memory Manager (Algorithm 5) with the frozen Answer Agent.
# ----------------------------------------------------------------------------

echo "Starting MemR1 Training..."
echo "Model: $MODEL_NAME"
echo "Answer base model: $MODEL_PATH"
echo "Answer adapter: $ANSWER_ADAPTER_PATH"

python3 scripts/train_manager_grpo.py \
    --manager-model "$MODEL_PATH" \
    --answer-model "$MODEL_PATH" \
    --answer-adapter "$ANSWER_ADAPTER_PATH" \
    --data-path "$DATA_PATH" \
    --validation-timelines "$VALIDATION_TIMELINES" \
    --output-dir "$OUTPUT_DIR" \
    --learning-rate "$LR" \
    --beta "$BETA" \
    --max-new-tokens "$MAX_GEN_LEN" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --num-generations "$NUM_GENS" \
    --answer-top-k "$ANSWER_TOP_K_PER_SPEAKER" \
    --device "cuda" \
    --epochs "$EPOCHS" \
    --seed "${SEED:-42}" \
    --reward "$QA_REWARD" \
    --answer-temperature 0.0 \
    --evidence-weight "$EVIDENCE_WEIGHT" \
    --terminal-potential "$TERMINAL_POTENTIAL" \
    --gamma "$GAMMA"
