#!/bin/bash
set -euo pipefail

# Flush Python output immediately when stdout is redirected by nohup.
export PYTHONUNBUFFERED=1

# Frozen three-stage QLoRA pipeline:
#   0. deterministic Stage-0 states/timelines + passed counterfactual gate
#   1. single-step Manager GRPO, checkpoint selected by roll-forward val F1
#   2. selected Manager builds Answer train/validation tuples
#   3. Answer GRPO

RAW_DATA_PATH="${RAW_DATA_PATH:-./data/locomo10.json}"
FACTS_DATA_PATH="${FACTS_DATA_PATH:-./data/manager_training_data_vllm.jsonl}"
HISTORY_WINDOW="${HISTORY_WINDOW:-fullprefix}"
case "$HISTORY_WINDOW" in
    fullprefix) WINDOW_SUFFIX="" ;;
    24) WINDOW_SUFFIX="_window24" ;;
    *) echo "HISTORY_WINDOW must be fullprefix or 24" >&2; exit 1 ;;
esac
STAGE0_DIR="${STAGE0_DIR:-./data/stage0${WINDOW_SUFFIX}}"
PRECHECK_PATH="${PRECHECK_PATH:-${STAGE0_DIR}/precheck_train_f1.json}"

MODEL_NAME="${MODEL_NAME:-Qwen2.5-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/models/${MODEL_NAME}}"
MANAGER_OUTPUT_DIR="${MANAGER_OUTPUT_DIR:-./output/mem_r1_manager${WINDOW_SUFFIX}_${MODEL_NAME}}"
ANSWER_OUTPUT_DIR="${ANSWER_OUTPUT_DIR:-./output/mem_r1_answer${WINDOW_SUFFIX}_${MODEL_NAME}}"
ANSWER_TRAIN_PATH="${ANSWER_TRAIN_PATH:-./data/answer_training${WINDOW_SUFFIX}_${MODEL_NAME}_train.jsonl}"
ANSWER_VAL_PATH="${ANSWER_VAL_PATH:-./data/answer_training${WINDOW_SUFFIX}_${MODEL_NAME}_validation.jsonl}"

MANAGER_EPOCHS="${MANAGER_EPOCHS:-5}"
ANSWER_EPOCHS="${ANSWER_EPOCHS:-5}"
MANAGER_LR="${MANAGER_LR:-1e-5}"
ANSWER_LR="${ANSWER_LR:-1e-5}"
BETA="${BETA:-0.01}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MANAGER_NUM_GENS="${MANAGER_NUM_GENS:-4}"
ANSWER_NUM_GENS="${ANSWER_NUM_GENS:-4}"
SAVE_EVERY_GROUPS="${SAVE_EVERY_GROUPS:-10}"
SCORE_MICRO_BATCH="${SCORE_MICRO_BATCH:-2}"
NUM_GPUS="${NUM_GPUS:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
RESUME_MANAGER_FROM="${RESUME_MANAGER_FROM:-}"
RESUME_ANSWER_FROM="${RESUME_ANSWER_FROM:-}"

cd "$(dirname "$0")/.."

if [ "$NUM_GPUS" -gt 1 ]; then
    TRAIN_LAUNCH=(
        accelerate launch --multi_gpu
        --num_processes "$NUM_GPUS"
        --num_machines 1
        --mixed_precision no
        --dynamo_backend no
        --main_process_port "$MAIN_PROCESS_PORT"
    )
else
    TRAIN_LAUNCH=(python3)
fi

if [ ! -f "$STAGE0_DIR/manager_states_train.jsonl" ] || \
   [ ! -f "$STAGE0_DIR/manager_states_validation.jsonl" ] || \
   [ ! -f "$STAGE0_DIR/dialogue_timelines_train.json" ] || \
   [ ! -f "$STAGE0_DIR/dialogue_timelines_validation.json" ] || \
   [ -n "${FORCE_STAGE0_REBUILD:-}" ]; then
    python3 scripts/rebuild_manager_states.py \
        --raw "$RAW_DATA_PATH" \
        --facts-jsonl "$FACTS_DATA_PATH" \
        --output-dir "$STAGE0_DIR" \
        --splits train validation --history-window "$HISTORY_WINDOW"
fi

# Reject accidentally reusing fullprefix states for the optional window24 run.
python3 -c 'import json,sys; expected=None if sys.argv[1]=="fullprefix" else 24; rows=[json.loads(line) for p in sys.argv[2:] for line in open(p,encoding="utf-8") if line.strip()]; assert rows and all(r.get("history_window")==expected and all((max(0,r["target_turn"]-expected) if expected else 0)<=m["source_turn"]<r["target_turn"] for m in r["old_bank"]) for r in rows), "Stage-0 history mismatch; rebuild into a separate directory and rerun precheck"' "$HISTORY_WINDOW" "$STAGE0_DIR/manager_states_train.jsonl" "$STAGE0_DIR/manager_states_validation.jsonl"

if [ ! -f "$PRECHECK_PATH" ]; then
    echo "Missing Stage-0.5 precheck: $PRECHECK_PATH" >&2
    echo "Run precheck_counterfactual.py and pass the gate before training." >&2
    exit 1
fi
python3 -c 'import json,sys; s=json.load(open(sys.argv[1],encoding="utf-8"))["summary"]; assert s.get("training_reward_rule")=="continuous_token_f1_unique_overlap", "stale precheck reward rule"; assert s.get("gate_passed"), "counterfactual gate failed"' "$PRECHECK_PATH"

MANAGER_ARGS=(
    --manager-model "$MODEL_PATH"
    --answer-model "$MODEL_PATH"
    --data-path "$STAGE0_DIR/manager_states_train.jsonl"
    --validation-states "$STAGE0_DIR/manager_states_validation.jsonl"
    --validation-timelines "$STAGE0_DIR/dialogue_timelines_validation.json"
    --precheck-path "$PRECHECK_PATH"
    --output-dir "$MANAGER_OUTPUT_DIR"
    --learning-rate "$MANAGER_LR"
    --beta "$BETA"
    --max-prompt-tokens "$MAX_PROMPT_TOKENS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --answer-max-new-tokens "$MAX_NEW_TOKENS"
    --num-generations "$MANAGER_NUM_GENS"
    --save-every-groups "$SAVE_EVERY_GROUPS"
    --epochs "$MANAGER_EPOCHS"
)
if [ -n "$RESUME_MANAGER_FROM" ]; then
    MANAGER_ARGS+=(--resume-from "$RESUME_MANAGER_FROM")
fi

if [ ! -f "$MANAGER_OUTPUT_DIR/best_checkpoint.json" ] || \
   [ -n "${FORCE_MANAGER_RETRAIN:-}" ]; then
    "${TRAIN_LAUNCH[@]}" scripts/train_manager_grpo.py "${MANAGER_ARGS[@]}"
fi

BEST_MANAGER_ADAPTER="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["path"])' "$MANAGER_OUTPUT_DIR/best_checkpoint.json")"
if [ ! -f "$BEST_MANAGER_ADAPTER/adapter_config.json" ]; then
    echo "Selected Manager adapter is missing: $BEST_MANAGER_ADAPTER" >&2
    exit 1
fi

if [ ! -f "$ANSWER_TRAIN_PATH" ] || [ -n "${FORCE_ANSWER_DATA:-}" ]; then
    python3 scripts/build_answer_training_data.py \
        --timeline "$STAGE0_DIR/dialogue_timelines_train.json" \
        --output "$ANSWER_TRAIN_PATH" \
        --manager-model "$MODEL_PATH" --manager-adapter "$BEST_MANAGER_ADAPTER" \
        --device cuda --manager-top-k 5 --answer-top-k-per-speaker 30 \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --max-new-tokens "$MAX_NEW_TOKENS"
fi
if [ ! -f "$ANSWER_VAL_PATH" ] || [ -n "${FORCE_ANSWER_DATA:-}" ]; then
    python3 scripts/build_answer_training_data.py \
        --timeline "$STAGE0_DIR/dialogue_timelines_validation.json" \
        --output "$ANSWER_VAL_PATH" \
        --manager-model "$MODEL_PATH" --manager-adapter "$BEST_MANAGER_ADAPTER" \
        --device cuda --manager-top-k 5 --answer-top-k-per-speaker 30 \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --max-new-tokens "$MAX_NEW_TOKENS"
fi

if [ ! -f "$ANSWER_OUTPUT_DIR/best_checkpoint.json" ] || \
   [ -n "${FORCE_ANSWER_RETRAIN:-}" ]; then
    ANSWER_RESUME_ARGS=()
    if [ -n "$RESUME_ANSWER_FROM" ]; then
        ANSWER_RESUME_ARGS+=(--resume-from "$RESUME_ANSWER_FROM")
    fi
    "${TRAIN_LAUNCH[@]}" scripts/train_answer_grpo.py \
        --data-path "$ANSWER_TRAIN_PATH" \
        --validation-data-path "$ANSWER_VAL_PATH" \
        --model-path "$MODEL_PATH" \
        --output-dir "$ANSWER_OUTPUT_DIR" \
        --learning-rate "$ANSWER_LR" \
        --beta "$BETA" \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --num-generations "$ANSWER_NUM_GENS" \
        --score-micro-batch "$SCORE_MICRO_BATCH" \
        --split all --epochs "$ANSWER_EPOCHS" \
        "${ANSWER_RESUME_ARGS[@]}"
fi

echo "Training pipeline complete."
echo "Manager: $BEST_MANAGER_ADAPTER"
if [ -f "$ANSWER_OUTPUT_DIR/best_checkpoint.json" ]; then
    BEST_ANSWER_ADAPTER="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["path"])' "$ANSWER_OUTPUT_DIR/best_checkpoint.json")"
else
    BEST_ANSWER_ADAPTER="$ANSWER_OUTPUT_DIR/epoch_${ANSWER_EPOCHS}"
fi
echo "Answer: $BEST_ANSWER_ADAPTER"
