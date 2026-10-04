#!/bin/bash
set -x

# ----------------------------------------------------------------------------
# Configuration for MemR1 Training
# ----------------------------------------------------------------------------

# 1. Paths
# DATA_PATH must be Algorithm 1 tuples (build_manager_training_data.py
# output): per-turn episodes with a 24-turn dialogue window.
DATA_PATH="${DATA_PATH:-./data/manager_training_data.jsonl}"
RAW_DATA_PATH="${RAW_DATA_PATH:-./data/locomo10.json}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/models/${MODEL_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/mem_r1_lora_${MODEL_NAME}}"

# Answer Agent (paper Section 3.3) training data: Algorithm 2 tuples
# (build_answer_training_data.py output over a Manager-generated memory bank).
ANSWER_DATA_PATH="${ANSWER_DATA_PATH:-./data/answer_training_data.jsonl}"
ANSWER_OUTPUT_DIR="${ANSWER_OUTPUT_DIR:-./output/mem_r1_lora_answer_${MODEL_NAME}}"
ANSWER_MANAGER_PATH="${ANSWER_MANAGER_PATH:-$MODEL_PATH}"
# Optional explicit override; otherwise the last Answer Agent adapter is used.
ANSWER_ADAPTER_PATH="${ANSWER_ADAPTER_PATH:-}"

# 2. Training Hyperparameters
LR="${LR:-1e-6}"
BETA="${BETA:-0.01}"
ANSWER_MAX_GEN_LEN="${ANSWER_MAX_GEN_LEN:-2048}"
MANAGER_MAX_GEN_LEN="${MANAGER_MAX_GEN_LEN:-2048}"
ANSWER_NUM_GENS="${ANSWER_NUM_GENS:-8}"
MANAGER_NUM_GENS="${MANAGER_NUM_GENS:-8}"
SCORE_MICRO_BATCH="${SCORE_MICRO_BATCH:-1}"
EPOCHS="${EPOCHS:-5}"
NUM_GPUS="${NUM_GPUS:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

# ----------------------------------------------------------------------------
# 1) Train the Answer Agent (Section 3.3) on Algorithm 2 tuples.
#    Algorithm 5 requires a fixed Answer Agent as the reward oracle.
# ----------------------------------------------------------------------------

cd "$(dirname "$0")/.." # Go to project root

if [ "$NUM_GPUS" -gt 1 ]; then
    TRAIN_LAUNCH=(
        accelerate launch
        --multi_gpu
        --num_processes "$NUM_GPUS"
        --num_machines 1
        --mixed_precision no
        --dynamo_backend no
        --main_process_port "$MAIN_PROCESS_PORT"
    )
else
    TRAIN_LAUNCH=(python3)
fi

if [ -z "$ANSWER_ADAPTER_PATH" ]; then
    if [ -f "$ANSWER_OUTPUT_DIR/epoch_${EPOCHS}/adapter_config.json" ] && [ -z "$FORCE_ANSWER_RETRAIN" ]; then
        echo "Answer Agent adapter found: $ANSWER_OUTPUT_DIR/epoch_${EPOCHS}"
    else
        if [ ! -f "$ANSWER_DATA_PATH" ]; then
            python3 scripts/build_answer_training_data.py \
                --input "$RAW_DATA_PATH" \
                --output "$ANSWER_DATA_PATH" \
                --manager-model "$ANSWER_MANAGER_PATH" \
                --device "cuda" \
                --manager-top-k 5 \
                --answer-top-k-per-speaker 30 \
                --max-new-tokens "$MANAGER_MAX_GEN_LEN" \
                --split train
        fi
        echo "Training Answer Agent on $ANSWER_DATA_PATH ..."
        "${TRAIN_LAUNCH[@]}" scripts/train_answer_grpo.py \
            --data-path "$ANSWER_DATA_PATH" \
            --raw-data-path "$RAW_DATA_PATH" \
            --model-path "$MODEL_PATH" \
            --output-dir "$ANSWER_OUTPUT_DIR" \
            --learning-rate "$LR" \
            --beta "$BETA" \
            --max-prompt-tokens 4096 \
            --max-new-tokens "$ANSWER_MAX_GEN_LEN" \
            --num-generations "$ANSWER_NUM_GENS" \
            --score-micro-batch "$SCORE_MICRO_BATCH" \
            --epochs "$EPOCHS"
    fi
    ANSWER_ADAPTER_PATH="$ANSWER_OUTPUT_DIR/epoch_${EPOCHS}"
fi

# ----------------------------------------------------------------------------
# 2) Train the Memory Manager (Algorithm 5) with the frozen Answer Agent.
# ----------------------------------------------------------------------------

echo "Starting MemR1 Training..."
echo "Model: $MODEL_NAME"
echo "GPUs: $NUM_GPUS"
echo "Answer base model: $MODEL_PATH"
echo "Answer adapter: $ANSWER_ADAPTER_PATH"

"${TRAIN_LAUNCH[@]}" scripts/train_manager_grpo.py \
    --manager-model "$MODEL_PATH" \
    --answer-model "$MODEL_PATH" \
    --answer-adapter "$ANSWER_ADAPTER_PATH" \
    --data-path "$DATA_PATH" \
    --raw-data-path "$RAW_DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --learning-rate "$LR" \
    --beta "$BETA" \
    --max-prompt-tokens 4096 \
    --max-new-tokens "$MANAGER_MAX_GEN_LEN" \
    --num-generations "$MANAGER_NUM_GENS" \
    --epochs "$EPOCHS"
