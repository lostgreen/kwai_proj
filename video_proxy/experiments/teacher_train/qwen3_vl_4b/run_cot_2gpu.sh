#!/usr/bin/env bash
# Convert an existing Qwen3-VL teacher experiment to CoT prompts, then run a 2-GPU smoke train.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL_FAMILY="qwen3_vl"
MODEL_SIZE="4b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"

MULTI_TASK_DATA_ROOT="${MULTI_TASK_DATA_ROOT:-${THREE_TASK_DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task}}"
EXPERIMENTS_DIR="${MULTI_TASK_DATA_ROOT}/experiments"
SOURCE_EXP_NAME="${SOURCE_EXP_NAME:-composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema}"
SOURCE_TRAIN_FILE="${SOURCE_TRAIN_FILE:-${EXPERIMENTS_DIR}/${SOURCE_EXP_NAME}/train.jsonl}"
SOURCE_TEST_FILE="${SOURCE_TEST_FILE:-${EXPERIMENTS_DIR}/${SOURCE_EXP_NAME}/val.jsonl}"

EXP_NAME="${EXP_NAME:-qwen3_vl_4b_teacher_cot_2gpu_smoke}"
EXP_DATA_DIR="${EXPERIMENTS_DIR}/${EXP_NAME}"
TRAIN_FILE="${TRAIN_FILE:-${EXP_DATA_DIR}/train.jsonl}"
TEST_FILE="${TEST_FILE:-${EXP_DATA_DIR}/val.jsonl}"

REASONING_TAG="${REASONING_TAG:-think}"
SAMPLE_PROMPTS_PER_TYPE="${SAMPLE_PROMPTS_PER_TYPE:-2}"
SAMPLE_PROMPT_MAX_CHARS="${SAMPLE_PROMPT_MAX_CHARS:-1600}"
CONVERT_FORCE="${CONVERT_FORCE:-false}"

if [[ ! -f "${SOURCE_TRAIN_FILE}" ]]; then
    echo "[qwen3-cot-2gpu] ERROR: SOURCE_TRAIN_FILE not found: ${SOURCE_TRAIN_FILE}" >&2
    echo "[qwen3-cot-2gpu] Set SOURCE_EXP_NAME or SOURCE_TRAIN_FILE to the previous no-CoT experiment data." >&2
    exit 1
fi
if [[ ! -f "${SOURCE_TEST_FILE}" ]]; then
    echo "[qwen3-cot-2gpu] ERROR: SOURCE_TEST_FILE not found: ${SOURCE_TEST_FILE}" >&2
    echo "[qwen3-cot-2gpu] Set SOURCE_EXP_NAME or SOURCE_TEST_FILE to the previous no-CoT experiment data." >&2
    exit 1
fi

mkdir -p "${EXP_DATA_DIR}"
if [[ ! -f "${TRAIN_FILE}" || "${CONVERT_FORCE,,}" =~ ^(true|1|yes)$ ]]; then
    python3 "${REPO_ROOT}/video_proxy/data/scripts/convert_jsonl_to_cot.py" \
        "${SOURCE_TRAIN_FILE}" "${TRAIN_FILE}" \
        --reasoning-tag "${REASONING_TAG}" \
        --sample-prompts "${SAMPLE_PROMPTS_PER_TYPE}" \
        --sample-prompt-max-chars "${SAMPLE_PROMPT_MAX_CHARS}"
else
    echo "[qwen3-cot-2gpu] Reusing existing CoT train file: ${TRAIN_FILE}"
    python3 "${REPO_ROOT}/video_proxy/data/scripts/convert_jsonl_to_cot.py" \
        "${TRAIN_FILE}" "${TRAIN_FILE}.sample_check.jsonl" \
        --reasoning-tag "${REASONING_TAG}" \
        --sample-prompts "${SAMPLE_PROMPTS_PER_TYPE}" \
        --sample-prompt-max-chars "${SAMPLE_PROMPT_MAX_CHARS}"
    rm -f "${TRAIN_FILE}.sample_check.jsonl"
fi

if [[ ! -f "${TEST_FILE}" || "${CONVERT_FORCE,,}" =~ ^(true|1|yes)$ ]]; then
    python3 "${REPO_ROOT}/video_proxy/data/scripts/convert_jsonl_to_cot.py" \
        "${SOURCE_TEST_FILE}" "${TEST_FILE}" \
        --reasoning-tag "${REASONING_TAG}" \
        --sample-prompts "${SAMPLE_PROMPTS_PER_TYPE}" \
        --sample-prompt-max-chars "${SAMPLE_PROMPT_MAX_CHARS}"
else
    echo "[qwen3-cot-2gpu] Reusing existing CoT val file: ${TEST_FILE}"
fi

TASKS="${TASKS:-tg mcq hier_seg event_logic aot}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
TP_SIZE="${TP_SIZE:-1}"
ROLLOUT_BS="${ROLLOUT_BS:-2}"
GLOBAL_BS="${GLOBAL_BS:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
ROLLOUT_N="${ROLLOUT_N:-1}"
MAX_STEPS="${MAX_STEPS:-1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
VAL_FREQ="${VAL_FREQ:--1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
SAVE_BEST="${SAVE_BEST:-false}"
SAVE_LIMIT="${SAVE_LIMIT:--1}"
ENABLE_GPU_FILLER="${ENABLE_GPU_FILLER:-false}"
POST_TRAIN_OCCUPANCY="${POST_TRAIN_OCCUPANCY:-false}"
CHECK_EXPERIMENT_JSONL="${CHECK_EXPERIMENT_JSONL:-true}"
CHECK_EXPERIMENT_FRAME_FILES="${CHECK_EXPERIMENT_FRAME_FILES:-false}"
MIX_FORCE="${MIX_FORCE:-false}"
REUSE_EXISTING_DATA="${REUSE_EXISTING_DATA:-true}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-20480}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
COT_BUDGET_ENABLED="${COT_BUDGET_ENABLED:-true}"
COT_BUDGET_START_TOKEN="${COT_BUDGET_START_TOKEN:-<${REASONING_TAG}>}"
COT_BUDGET_END_TOKEN="${COT_BUDGET_END_TOKEN:-</${REASONING_TAG}>}"
COT_BUDGET_MAX_TOKENS="${COT_BUDGET_MAX_TOKENS:-128}"

export MULTI_TASK_DATA_ROOT TRAIN_FILE TEST_FILE
export TASKS N_GPUS_PER_NODE TP_SIZE ROLLOUT_BS GLOBAL_BS VAL_BATCH_SIZE ROLLOUT_N MAX_STEPS
export VAL_BEFORE_TRAIN VAL_FREQ SAVE_FREQ SAVE_BEST SAVE_LIMIT
export ENABLE_GPU_FILLER POST_TRAIN_OCCUPANCY CHECK_EXPERIMENT_JSONL CHECK_EXPERIMENT_FRAME_FILES MIX_FORCE
export REUSE_EXISTING_DATA
export ROLLOUT_MAX_NUM_SEQS ROLLOUT_MAX_BATCHED_TOKENS MAX_RESPONSE_LEN
export COT_BUDGET_ENABLED COT_BUDGET_START_TOKEN COT_BUDGET_END_TOKEN COT_BUDGET_MAX_TOKENS

source "${REPO_ROOT}/video_proxy/training/recipes/teacher_train_ema_grpo.sh"

ROLLOUT_JSONL="${CHECKPOINT_ROOT}/${EXP_NAME}/rollouts/step_000001.jsonl"
if [[ -f "${ROLLOUT_JSONL}" ]]; then
    python3 "${REPO_ROOT}/video_proxy/training/tools/check_cot_budget_rollout.py" \
        "${ROLLOUT_JSONL}" \
        --start-token "${COT_BUDGET_START_TOKEN}" \
        --end-token "${COT_BUDGET_END_TOKEN}" \
        --max-tokens "${COT_BUDGET_MAX_TOKENS}" \
        --tokenizer "${MODEL_PATH}" \
        --require-start \
        --require-closed \
        --require-within-budget
else
    echo "[qwen3-cot-2gpu] WARN: rollout file not found after smoke run: ${ROLLOUT_JSONL}" >&2
fi
