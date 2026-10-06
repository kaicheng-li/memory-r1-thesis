#!/bin/bash
set -e -x

# Memory-R1 QLoRA pipeline:
# 1. Bootstrap Manager with a frozen base Answer Agent.
# 2. Rebuild Answer tuples with the bootstrap Manager adapter.
# 3. Train the Answer Agent.
# 4. Train the final Manager with the frozen Answer adapter.

DATA_PATH="${DATA_PATH:-./data/manager_training_data_vllm.jsonl}"
RAW_DATA_PATH="${RAW_DATA_PATH:-./data/locomo10.json}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/models/${MODEL_NAME}}"

BOOTSTRAP_MANAGER_DIR="${BOOTSTRAP_MANAGER_DIR:-./output/mem_r1_bootstrap_manager_${MODEL_NAME}}"
BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-1}"

ANSWER_DATA_PATH="${ANSWER_DATA_PATH:-./data/answer_training_data_bootstrap_${MODEL_NAME}.jsonl}"
ANSWER_OUTPUT_DIR="${ANSWER_OUTPUT_DIR:-./output/mem_r1_answer_${MODEL_NAME}}"
ANSWER_EPOCHS="${ANSWER_EPOCHS:-1}"
ANSWER_ADAPTER_PATH="${ANSWER_ADAPTER_PATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-./output/mem_r1_manager_${MODEL_NAME}}"
EPOCHS="${EPOCHS:-5}"

LR="${LR:-1e-6}"
BETA="${BETA:-0.01}"
# Manager and Answer prompts request compact structured/short answers.  Keeping
# the cap finite prevents one malformed rollout from blocking all ranks for
# the NCCL heartbeat window.
ANSWER_MAX_GEN_LEN="${ANSWER_MAX_GEN_LEN:-512}"
MANAGER_MAX_GEN_LEN="${MANAGER_MAX_GEN_LEN:-512}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
ANSWER_NUM_GENS="${ANSWER_NUM_GENS:-4}"
MANAGER_NUM_GENS="${MANAGER_NUM_GENS:-4}"
SCORE_MICRO_BATCH="${SCORE_MICRO_BATCH:-2}"
NUM_GPUS="${NUM_GPUS:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

cd "$(dirname "$0")/.."

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

BOOTSTRAP_MANAGER_ADAPTER="$BOOTSTRAP_MANAGER_DIR/epoch_${BOOTSTRAP_EPOCHS}"
if [ ! -f "$BOOTSTRAP_MANAGER_ADAPTER/adapter_config.json" ] || [ -n "${FORCE_BOOTSTRAP_MANAGER:-}" ]; then
    echo "Bootstrapping Manager with frozen base Answer Agent ..."
    "${TRAIN_LAUNCH[@]}" scripts/train_manager_grpo.py \
        --manager-model "$MODEL_PATH" \
        --answer-model "$MODEL_PATH" \
        --data-path "$DATA_PATH" \
        --raw-data-path "$RAW_DATA_PATH" \
        --output-dir "$BOOTSTRAP_MANAGER_DIR" \
        --learning-rate "$LR" \
        --beta "$BETA" \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --max-new-tokens "$MANAGER_MAX_GEN_LEN" \
        --answer-max-new-tokens "$ANSWER_MAX_GEN_LEN" \
        --num-generations "$MANAGER_NUM_GENS" \
        --epochs "$BOOTSTRAP_EPOCHS"
fi

if [ ! -f "$ANSWER_DATA_PATH" ] || [ -n "${FORCE_ANSWER_DATA:-}" ]; then
    echo "Building Answer data with $BOOTSTRAP_MANAGER_ADAPTER ..."
    python3 scripts/build_answer_training_data.py \
        --input "$RAW_DATA_PATH" \
        --output "$ANSWER_DATA_PATH" \
        --manager-model "$MODEL_PATH" \
        --manager-adapter "$BOOTSTRAP_MANAGER_ADAPTER" \
        --device cuda \
        --manager-top-k 5 \
        --answer-top-k-per-speaker 30 \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --max-new-tokens "$MANAGER_MAX_GEN_LEN" \
        --split train
fi

if [ -z "$ANSWER_ADAPTER_PATH" ]; then
    ANSWER_ADAPTER_PATH="$ANSWER_OUTPUT_DIR/epoch_${ANSWER_EPOCHS}"
    if [ ! -f "$ANSWER_ADAPTER_PATH/adapter_config.json" ] || [ -n "${FORCE_ANSWER_RETRAIN:-}" ]; then
        echo "Training Answer Agent on $ANSWER_DATA_PATH ..."
        "${TRAIN_LAUNCH[@]}" scripts/train_answer_grpo.py \
            --data-path "$ANSWER_DATA_PATH" \
            --model-path "$MODEL_PATH" \
            --output-dir "$ANSWER_OUTPUT_DIR" \
            --learning-rate "$LR" \
            --beta "$BETA" \
            --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
            --max-new-tokens "$ANSWER_MAX_GEN_LEN" \
            --num-generations "$ANSWER_NUM_GENS" \
            --score-micro-batch "$SCORE_MICRO_BATCH" \
            --split all \
            --epochs "$ANSWER_EPOCHS"
    fi
fi

FINAL_MANAGER_ADAPTER="$OUTPUT_DIR/epoch_${EPOCHS}"
if [ ! -f "$FINAL_MANAGER_ADAPTER/adapter_config.json" ] || [ -n "${FORCE_MANAGER_RETRAIN:-}" ]; then
    echo "Training final Manager with frozen Answer adapter $ANSWER_ADAPTER_PATH ..."
    "${TRAIN_LAUNCH[@]}" scripts/train_manager_grpo.py \
        --manager-model "$MODEL_PATH" \
        --answer-model "$MODEL_PATH" \
        --answer-adapter "$ANSWER_ADAPTER_PATH" \
        --data-path "$DATA_PATH" \
        --raw-data-path "$RAW_DATA_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --learning-rate "$LR" \
        --beta "$BETA" \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --max-new-tokens "$MANAGER_MAX_GEN_LEN" \
        --answer-max-new-tokens "$ANSWER_MAX_GEN_LEN" \
        --num-generations "$MANAGER_NUM_GENS" \
        --epochs "$EPOCHS"
fi
