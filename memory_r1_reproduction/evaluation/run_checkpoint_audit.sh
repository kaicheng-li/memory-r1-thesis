#!/bin/bash
set -euo pipefail

# Full-data checkpoint audit. Each checkpoint is evaluated in its own process;
# AUDIT_GPUS controls round-robin GPU assignment. Existing reports are skipped.

cd "$(dirname "$0")/.."

MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/models/${MODEL_NAME}}"
MANAGER_DATA_PATH="${MANAGER_DATA_PATH:-./data/manager_training_data_vllm.jsonl}"
RAW_DATA_PATH="${RAW_DATA_PATH:-./data/locomo10.json}"
AUDIT_SPLIT="${AUDIT_SPLIT:-train}"
ANSWER_DATA_PATH="${ANSWER_DATA_PATH:-./data/answer_training_data_bootstrap_${MODEL_NAME}.jsonl}"
BOOTSTRAP_ADAPTER="${BOOTSTRAP_ADAPTER:-./output/mem_r1_bootstrap_manager_${MODEL_NAME}/epoch_1}"
ANSWER_ADAPTER="${ANSWER_ADAPTER:-./output/mem_r1_answer_${MODEL_NAME}/epoch_1}"
MANAGER_DIR="${MANAGER_DIR:-./output/mem_r1_manager_${MODEL_NAME}}"
# Manager epochs to audit, as a comma-separated list. The baseline comparisons
# below are always included; saved Manager checkpoints default to epochs 4 and 5.
MANAGER_EPOCHS="${MANAGER_EPOCHS:-4,5}"

# Full QA coverage, one deterministic generation per checkpoint. Set these to
# 4 and 1.0 only when reproducing the stochastic training rollout distribution.
ANSWER_SAMPLE_COUNT="${ANSWER_SAMPLE_COUNT:-0}"
MANAGER_SAMPLE_COUNT="${MANAGER_SAMPLE_COUNT:-0}"
ANSWER_NUM_GENS="${ANSWER_NUM_GENS:-1}"
MANAGER_NUM_GENS="${MANAGER_NUM_GENS:-1}"
ANSWER_TEMPERATURE="${ANSWER_TEMPERATURE:-0.0}"
MANAGER_TEMPERATURE="${MANAGER_TEMPERATURE:-0.0}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SEED="${SEED:-42}"
AUDIT_GPUS="${AUDIT_GPUS:-0,1,2,3}"
AUDIT_PROFILE="${AUDIT_PROFILE:-a-g${ANSWER_NUM_GENS}-t${ANSWER_TEMPERATURE}_m-g${MANAGER_NUM_GENS}-t${MANAGER_TEMPERATURE}}"
AUDIT_DIR="${AUDIT_DIR:-./audit/checkpoints_${MODEL_NAME}_${AUDIT_SPLIT}_${AUDIT_PROFILE}}"
AUDIT_LOG_DIR="${AUDIT_LOG_DIR:-./logs/checkpoint_audit_${MODEL_NAME}_${AUDIT_SPLIT}_${AUDIT_PROFILE}}"

mkdir -p "$AUDIT_DIR" "$AUDIT_LOG_DIR"
IFS=',' read -ra GPU_LIST <<< "$AUDIT_GPUS"
if [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "AUDIT_GPUS must contain at least one GPU id" >&2
    exit 1
fi

run_answer_job() {
    LAST_JOB_PID=""
    local gpu="$1"
    local label="$2"
    local adapter="${3:-}"
    local output="$AUDIT_DIR/answer_${label}.json"
    local log="$AUDIT_LOG_DIR/answer_${label}.log"
    if [ -f "$output" ]; then
        echo "Skipping existing $output"
        return 0
    fi
    local adapter_args=()
    if [ -n "$adapter" ]; then
        if [ ! -f "$adapter/adapter_config.json" ]; then
            echo "Missing Answer adapter: $adapter" >&2
            return 1
        fi
        adapter_args=(--adapter "$adapter")
    fi
    echo "Starting Answer $label on GPU $gpu; log=$log"
    CUDA_VISIBLE_DEVICES="$gpu" python3 evaluation/audit_checkpoints.py answer \
        --label "$label" \
        --model-path "$MODEL_PATH" \
        --data-path "$ANSWER_DATA_PATH" \
        --raw-data-path "$RAW_DATA_PATH" \
        --split "$AUDIT_SPLIT" \
        --output "$output" \
        --sample-count "$ANSWER_SAMPLE_COUNT" \
        --num-generations "$ANSWER_NUM_GENS" \
        --temperature "$ANSWER_TEMPERATURE" \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --seed "$SEED" \
        "${adapter_args[@]}" >"$log" 2>&1 &
    LAST_JOB_PID="$!"
}

run_manager_job() {
    LAST_JOB_PID=""
    local gpu="$1"
    local label="$2"
    local adapter="${3:-}"
    local answer_adapter="${4:-}"
    local output="$AUDIT_DIR/manager_${label}.json"
    local log="$AUDIT_LOG_DIR/manager_${label}.log"
    if [ -f "$output" ]; then
        echo "Skipping existing $output"
        return 0
    fi
    local adapter_args=()
    local answer_adapter_args=()
    if [ -n "$adapter" ]; then
        if [ ! -f "$adapter/adapter_config.json" ]; then
            echo "Missing Manager adapter: $adapter" >&2
            return 1
        fi
        adapter_args=(--adapter "$adapter")
    fi
    if [ -n "$answer_adapter" ]; then
        if [ ! -f "$answer_adapter/adapter_config.json" ]; then
            echo "Missing Answer oracle adapter: $answer_adapter" >&2
            return 1
        fi
        answer_adapter_args=(--answer-adapter "$answer_adapter")
    fi
    echo "Starting Manager $label on GPU $gpu; log=$log"
    CUDA_VISIBLE_DEVICES="$gpu" python3 evaluation/audit_checkpoints.py manager \
        --label "$label" \
        --model-path "$MODEL_PATH" \
        --data-path "$MANAGER_DATA_PATH" \
        --raw-data-path "$RAW_DATA_PATH" \
        --split "$AUDIT_SPLIT" \
        --output "$output" \
        --sample-count "$MANAGER_SAMPLE_COUNT" \
        --num-generations "$MANAGER_NUM_GENS" \
        --temperature "$MANAGER_TEMPERATURE" \
        --answer-temperature 0.0 \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --answer-max-new-tokens "$MAX_NEW_TOKENS" \
        --seed "$SEED" \
        "${answer_adapter_args[@]}" \
        "${adapter_args[@]}" >"$log" 2>&1 &
    LAST_JOB_PID="$!"
}

wait_for_batch() {
    local failed=0
    local pid
    for pid in "$@"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "One or more audit jobs failed; inspect $AUDIT_LOG_DIR" >&2
        return 1
    fi
}

if [ ! -f "$ANSWER_DATA_PATH" ]; then
    echo "Missing Answer audit data: $ANSWER_DATA_PATH" >&2
    exit 1
fi
if [ ! -f "$ANSWER_ADAPTER/adapter_config.json" ]; then
    echo "Missing fixed Answer oracle adapter: $ANSWER_ADAPTER" >&2
    exit 1
fi

ANSWER_PIDS=()
run_answer_job "${GPU_LIST[0]}" base
if [ -n "$LAST_JOB_PID" ]; then ANSWER_PIDS+=("$LAST_JOB_PID"); fi
if [ "${#GPU_LIST[@]}" -gt 1 ]; then
    run_answer_job "${GPU_LIST[1]}" adapter "$ANSWER_ADAPTER"
    if [ -n "$LAST_JOB_PID" ]; then ANSWER_PIDS+=("$LAST_JOB_PID"); fi
else
    if [ "${#ANSWER_PIDS[@]}" -gt 0 ]; then wait_for_batch "${ANSWER_PIDS[@]}"; fi
    ANSWER_PIDS=()
    run_answer_job "${GPU_LIST[0]}" adapter "$ANSWER_ADAPTER"
    if [ -n "$LAST_JOB_PID" ]; then ANSWER_PIDS+=("$LAST_JOB_PID"); fi
fi
if [ "${#ANSWER_PIDS[@]}" -gt 0 ]; then wait_for_batch "${ANSWER_PIDS[@]}"; fi

MANAGER_LABELS=(
    base_base_oracle
    bootstrap_base_oracle
    base_final_oracle
    bootstrap_final_oracle
)
MANAGER_ADAPTERS=(
    ""
    "$BOOTSTRAP_ADAPTER"
    ""
    "$BOOTSTRAP_ADAPTER"
)
MANAGER_ANSWER_ADAPTERS=(
    ""
    ""
    "$ANSWER_ADAPTER"
    "$ANSWER_ADAPTER"
)

IFS=',' read -ra REQUESTED_EPOCHS <<< "$MANAGER_EPOCHS"
for epoch in "${REQUESTED_EPOCHS[@]}"; do
    if ! [[ "$epoch" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid MANAGER_EPOCHS entry: $epoch (expected comma-separated positive integers)" >&2
        exit 1
    fi
    MANAGER_LABELS+=("epoch_${epoch}_final_oracle")
    MANAGER_ADAPTERS+=("$MANAGER_DIR/epoch_${epoch}")
    MANAGER_ANSWER_ADAPTERS+=("$ANSWER_ADAPTER")
done

MANAGER_PIDS=()
for index in "${!MANAGER_LABELS[@]}"; do
    gpu="${GPU_LIST[${#MANAGER_PIDS[@]}]}"
    run_manager_job "$gpu" "${MANAGER_LABELS[$index]}" \
        "${MANAGER_ADAPTERS[$index]}" "${MANAGER_ANSWER_ADAPTERS[$index]}"
    if [ -n "$LAST_JOB_PID" ]; then MANAGER_PIDS+=("$LAST_JOB_PID"); fi
    if [ "${#MANAGER_PIDS[@]}" -eq "${#GPU_LIST[@]}" ]; then
        wait_for_batch "${MANAGER_PIDS[@]}"
        MANAGER_PIDS=()
    fi
done
if [ "${#MANAGER_PIDS[@]}" -gt 0 ]; then
    wait_for_batch "${MANAGER_PIDS[@]}"
fi

python3 evaluation/rescore_audit_metrics.py "$AUDIT_DIR" --raw-data-path "$RAW_DATA_PATH"
python3 evaluation/summarize_checkpoint_audit.py "$AUDIT_DIR"

if [ "${RUN_LLM_JUDGE:-0}" = "1" ]; then
    judge_args=(--backend "${JUDGE_BACKEND:-openai}")
    if [ -n "${JUDGE_MODEL:-}" ]; then judge_args+=(--model "$JUDGE_MODEL"); fi
    if [ -n "${JUDGE_BASE_URL:-}" ]; then judge_args+=(--base-url "$JUDGE_BASE_URL"); fi
    if [ -n "${JUDGE_API_KEY:-}" ]; then judge_args+=(--api-key "$JUDGE_API_KEY"); fi
    if [ -n "${JUDGE_LOCAL_MODEL_PATH:-}" ]; then
        judge_args+=(--local-model-path "$JUDGE_LOCAL_MODEL_PATH")
    fi
    judge_args+=(--max-workers "${JUDGE_MAX_WORKERS:-8}")
    python3 evaluation/score_audit_judge.py "$AUDIT_DIR" "${judge_args[@]}"
    python3 evaluation/summarize_checkpoint_audit.py "$AUDIT_DIR"
fi
