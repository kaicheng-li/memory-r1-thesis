#!/bin/bash
set -x

# ----------------------------------------------------------------------------
# Configuration for MemR1 Training
# ----------------------------------------------------------------------------

# 1. Paths
# DATA_PATH expects Algorithm 1 tuples (build_manager_training_data.py
# output, per-turn episodes with a 50-turn dialogue window). Raw LoCoMo JSON
# is accepted as a full-dialogue fallback.
DATA_PATH="${DATA_PATH:-./data/locomo_train.json}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-3B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/models/${MODEL_NAME}}"
ANSWER_MODEL_PATH="${ANSWER_MODEL_PATH:-$MODEL_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/mem_r1_qwen25_3b}"

# Answer Agent (paper Section 3.3) training data: Algorithm 2 tuples
# (build_answer_training_data.py output over the Algorithm 1 temporal banks).
ANSWER_DATA_PATH="${ANSWER_DATA_PATH:-./data/answer_training_data.jsonl}"
ANSWER_OUTPUT_DIR="${ANSWER_OUTPUT_DIR:-./output/mem_r1_answer_${MODEL_NAME}}"

# 2. Training Hyperparameters
LR=1e-6
BETA=0.01
MAX_GEN_LEN=256
NUM_GENS=8
EPOCHS=5

# ----------------------------------------------------------------------------
# 1) Train the Answer Agent (Section 3.3) if its Algorithm 2 data exists.
#    Algorithm 5 needs a fixed Answer Agent as the reward oracle.
# ----------------------------------------------------------------------------

cd "$(dirname "$0")/.." # Go to project root

if [ -f "$ANSWER_DATA_PATH" ] && { [ ! -d "$ANSWER_OUTPUT_DIR/epoch_${EPOCHS}" ] || [ -n "$FORCE_ANSWER_RETRAIN" ]; }; then
    echo "Training Answer Agent on $ANSWER_DATA_PATH ..."
    python3 scripts/train_answer_grpo.py \
        --data-path "$ANSWER_DATA_PATH" \
        --model-path "$MODEL_PATH" \
        --output-dir "$ANSWER_OUTPUT_DIR" \
        --learning-rate "$LR" \
        --beta "$BETA" \
        --max-new-tokens "$MAX_GEN_LEN" \
        --num-generations "$NUM_GENS" \
        --epochs "$EPOCHS"
fi
if [ -d "$ANSWER_OUTPUT_DIR/epoch_${EPOCHS}" ]; then
    ANSWER_MODEL_PATH="$ANSWER_OUTPUT_DIR/epoch_${EPOCHS}"
else
    echo "WARNING: Answer Agent data/checkpoint missing; using $MODEL_PATH as the frozen Answer Agent."
fi

# ----------------------------------------------------------------------------
# 2) Train the Memory Manager (Algorithm 5) with the frozen Answer Agent.
# ----------------------------------------------------------------------------

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
