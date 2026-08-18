#!/bin/bash
set -x

# ----------------------------------------------------------------------------
# Configuration for MemR1 Training
# ----------------------------------------------------------------------------

# 1. Paths
# Assuming same model and output structure
# DATA_PATH expects either Algorithm 1 tuples (build_manager_training_data.py
# output, per-turn episodes) or raw LoCoMo JSON (full-dialogue fallback).
DATA_PATH="${DATA_PATH:-./data/locomo_train.json}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/models/${MODEL_NAME}}"
ANSWER_MODEL_PATH="${ANSWER_MODEL_PATH:-$MODEL_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/mem_r1_qwen25_3b}"

# 2. Training Hyperparameters
LR=1e-6
BETA=0.01
MAX_GEN_LEN=256
NUM_GENS=8
EPOCHS=5

# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------

cd "$(dirname "$0")/.." # Go to project root

echo "Starting MemR1 Training..."
echo "Model: $MODEL_NAME"
echo "Answer model: $ANSWER_MODEL_PATH"

python3 scripts/train_manager_grpo.py \
    --manager-model "$MODEL_PATH" \
    --answer-model "$ANSWER_MODEL_PATH" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --learning-rate "$LR" \
    --beta "$BETA" \
    --max-new-tokens "$MAX_GEN_LEN" \
    --num-generations "$NUM_GENS" \
    --epochs "$EPOCHS"
