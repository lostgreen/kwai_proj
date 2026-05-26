#!/usr/bin/env bash
# Train one Qwen3-VL teacher from an existing mf256 experiment JSONL directory.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_FAMILY="${MODEL_FAMILY:?set MODEL_FAMILY before sourcing single_teacher_from_experiment.sh}"
MODEL_SIZE="${MODEL_SIZE:?set MODEL_SIZE before sourcing single_teacher_from_experiment.sh}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH before sourcing single_teacher_from_experiment.sh}"
TEACHER_KIND="${TEACHER_KIND:?set TEACHER_KIND before sourcing single_teacher_from_experiment.sh}"
SOURCE_EXP_NAME="${SOURCE_EXP_NAME:?set SOURCE_EXP_NAME before sourcing single_teacher_from_experiment.sh}"

MULTI_TASK_DATA_ROOT="${MULTI_TASK_DATA_ROOT:-${THREE_TASK_DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task}}"
EXPERIMENTS_DIR="${MULTI_TASK_DATA_ROOT}/experiments"
SOURCE_TRAIN_FILE="${SOURCE_TRAIN_FILE:-${EXPERIMENTS_DIR}/${SOURCE_EXP_NAME}/train.jsonl}"
SOURCE_TEST_FILE="${SOURCE_TEST_FILE:-${EXPERIMENTS_DIR}/${SOURCE_EXP_NAME}/val.jsonl}"

COT_MODE="${COT_MODE:-false}"
RUN_SUFFIX="${RUN_SUFFIX:-$([[ "${COT_MODE,,}" =~ ^(true|1|yes)$ ]] && echo cot || echo nocot)}"
EXP_NAME="${EXP_NAME:-${MODEL_FAMILY}_${MODEL_SIZE}_${TEACHER_KIND}_teacher_${RUN_SUFFIX}}"
EXP_DATA_DIR="${EXPERIMENTS_DIR}/${EXP_NAME}"
TRAIN_FILE="${TRAIN_FILE:-${EXP_DATA_DIR}/train.jsonl}"
TEST_FILE="${TEST_FILE:-${EXP_DATA_DIR}/val.jsonl}"

REASONING_TAG="${REASONING_TAG:-thought}"
SAMPLE_PROMPTS_PER_TYPE="${SAMPLE_PROMPTS_PER_TYPE:-2}"
SAMPLE_PROMPT_MAX_CHARS="${SAMPLE_PROMPT_MAX_CHARS:-1600}"
CONVERT_FORCE="${CONVERT_FORCE:-false}"
CONVERT_ONLY="${CONVERT_ONLY:-false}"

if [[ ! -f "${SOURCE_TRAIN_FILE}" ]]; then
    echo "[single-teacher] ERROR: SOURCE_TRAIN_FILE not found: ${SOURCE_TRAIN_FILE}" >&2
    echo "[single-teacher] Set SOURCE_EXP_NAME or SOURCE_TRAIN_FILE to an existing mf256 experiment JSONL." >&2
    exit 1
fi
if [[ ! -f "${SOURCE_TEST_FILE}" ]]; then
    echo "[single-teacher] ERROR: SOURCE_TEST_FILE not found: ${SOURCE_TEST_FILE}" >&2
    echo "[single-teacher] Set SOURCE_EXP_NAME or SOURCE_TEST_FILE to an existing mf256 experiment JSONL." >&2
    exit 1
fi

_SOURCE_FRAME_INFO="$(
python3 - "${SOURCE_TRAIN_FILE}" <<'PY'
import json
import sys

path = sys.argv[1]
policy = ""
max_frames = ""
version = ""
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sampling = (row.get("metadata") or {}).get("experiment_frame_sampling") or {}
        policy = str(sampling.get("policy") or "")
        value = sampling.get("max_frames")
        if value not in (None, ""):
            max_frames = str(int(value))
        version = str(sampling.get("implementation_version") or "")
        break
print(policy)
print(max_frames)
print(version)
PY
)"
_SOURCE_FRAME_SAMPLE_POLICY="$(printf '%s\n' "${_SOURCE_FRAME_INFO}" | sed -n '1p')"
_SOURCE_FRAME_SAMPLE_MAX_FRAMES="$(printf '%s\n' "${_SOURCE_FRAME_INFO}" | sed -n '2p')"
_SOURCE_FRAME_SAMPLE_POLICY_VERSION="$(printf '%s\n' "${_SOURCE_FRAME_INFO}" | sed -n '3p')"

mkdir -p "${EXP_DATA_DIR}"
if [[ "${COT_MODE,,}" =~ ^(true|1|yes)$ ]]; then
    if [[ ! -f "${TRAIN_FILE}" || "${CONVERT_FORCE,,}" =~ ^(true|1|yes)$ ]]; then
        python3 "${REPO_ROOT}/video_proxy/data/scripts/convert_jsonl_to_cot.py" \
            "${SOURCE_TRAIN_FILE}" "${TRAIN_FILE}" \
            --reasoning-tag "${REASONING_TAG}" \
            --sample-prompts "${SAMPLE_PROMPTS_PER_TYPE}" \
            --sample-prompt-max-chars "${SAMPLE_PROMPT_MAX_CHARS}"
    else
        echo "[single-teacher] Reusing existing CoT train file; normalizing in-place: ${TRAIN_FILE}"
        python3 "${REPO_ROOT}/video_proxy/data/scripts/convert_jsonl_to_cot.py" \
            "${TRAIN_FILE}" \
            --in-place \
            --reasoning-tag "${REASONING_TAG}" \
            --sample-prompts "${SAMPLE_PROMPTS_PER_TYPE}" \
            --sample-prompt-max-chars "${SAMPLE_PROMPT_MAX_CHARS}"
    fi

    if [[ ! -f "${TEST_FILE}" || "${CONVERT_FORCE,,}" =~ ^(true|1|yes)$ ]]; then
        python3 "${REPO_ROOT}/video_proxy/data/scripts/convert_jsonl_to_cot.py" \
            "${SOURCE_TEST_FILE}" "${TEST_FILE}" \
            --reasoning-tag "${REASONING_TAG}" \
            --sample-prompts "${SAMPLE_PROMPTS_PER_TYPE}" \
            --sample-prompt-max-chars "${SAMPLE_PROMPT_MAX_CHARS}"
    else
        echo "[single-teacher] Reusing existing CoT val file; normalizing in-place: ${TEST_FILE}"
        python3 "${REPO_ROOT}/video_proxy/data/scripts/convert_jsonl_to_cot.py" \
            "${TEST_FILE}" \
            --in-place \
            --reasoning-tag "${REASONING_TAG}" \
            --sample-prompts "${SAMPLE_PROMPTS_PER_TYPE}" \
            --sample-prompt-max-chars "${SAMPLE_PROMPT_MAX_CHARS}"
    fi
else
    TRAIN_FILE="${SOURCE_TRAIN_FILE}"
    TEST_FILE="${SOURCE_TEST_FILE}"
fi

echo "[single-teacher] Source train: ${SOURCE_TRAIN_FILE}"
echo "[single-teacher] Source val: ${SOURCE_TEST_FILE}"
echo "[single-teacher] Data ready: train=${TRAIN_FILE}"
echo "[single-teacher] Data ready: val=${TEST_FILE}"
if [[ -n "${_SOURCE_FRAME_SAMPLE_POLICY}" || -n "${_SOURCE_FRAME_SAMPLE_MAX_FRAMES}" ]]; then
    echo "[single-teacher] Inherited frame policy: policy=${_SOURCE_FRAME_SAMPLE_POLICY:-<empty>} max_frames=${_SOURCE_FRAME_SAMPLE_MAX_FRAMES:-<empty>} version=${_SOURCE_FRAME_SAMPLE_POLICY_VERSION:-<empty>}"
fi
if [[ "${CONVERT_ONLY,,}" =~ ^(true|1|yes)$ ]]; then
    echo "[single-teacher] CONVERT_ONLY=true; skip training."
    exit 0
fi

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
TP_SIZE="${TP_SIZE:-1}"
ROLLOUT_BS="${ROLLOUT_BS:-}"
if [[ -z "${ROLLOUT_BS}" ]]; then
    if [[ "${N_GPUS_PER_NODE}" == "8" ]]; then
        ROLLOUT_BS=64
    else
        ROLLOUT_BS="$((N_GPUS_PER_NODE * 4))"
    fi
fi
GLOBAL_BS="${GLOBAL_BS:-${ROLLOUT_BS}}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
if [[ -z "${VAL_BATCH_SIZE}" ]]; then
    if [[ "${N_GPUS_PER_NODE}" == "2" ]]; then
        VAL_BATCH_SIZE=32
    elif [[ "${N_GPUS_PER_NODE}" == "8" ]]; then
        VAL_BATCH_SIZE=128
    else
        VAL_BATCH_SIZE="${GLOBAL_BS}"
    fi
fi
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
LR="${LR:-5e-7}"
KL_COEF="${KL_COEF:-0.01}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.005}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
VAL_FREQ="${VAL_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
SAVE_BEST="${SAVE_BEST:-false}"
SAVE_LIMIT="${SAVE_LIMIT:--1}"
ENABLE_GPU_FILLER="${ENABLE_GPU_FILLER:-false}"
POST_TRAIN_OCCUPANCY="${POST_TRAIN_OCCUPANCY:-false}"
CHECK_EXPERIMENT_JSONL="${CHECK_EXPERIMENT_JSONL:-true}"
CHECK_EXPERIMENT_FRAME_FILES="${CHECK_EXPERIMENT_FRAME_FILES:-false}"
MIX_FORCE="${MIX_FORCE:-false}"
REUSE_EXISTING_DATA="${REUSE_EXISTING_DATA:-true}"
FRAME_SAMPLE_POLICY="${FRAME_SAMPLE_POLICY:-${_SOURCE_FRAME_SAMPLE_POLICY}}"
FRAME_SAMPLE_MAX_FRAMES="${FRAME_SAMPLE_MAX_FRAMES:-${_SOURCE_FRAME_SAMPLE_MAX_FRAMES}}"
FRAME_SAMPLE_POLICY_VERSION="${FRAME_SAMPLE_POLICY_VERSION:-${_SOURCE_FRAME_SAMPLE_POLICY_VERSION}}"
if [[ -n "${_SOURCE_FRAME_SAMPLE_MAX_FRAMES}" ]]; then
    MAX_FRAMES="${MAX_FRAMES:-${_SOURCE_FRAME_SAMPLE_MAX_FRAMES}}"
fi
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-20480}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"

if [[ "${COT_MODE,,}" =~ ^(true|1|yes)$ ]]; then
    COT_BUDGET_ENABLED="${COT_BUDGET_ENABLED:-true}"
    COT_BUDGET_START_TOKEN="${COT_BUDGET_START_TOKEN:-<${REASONING_TAG}>}"
    COT_BUDGET_END_TOKEN="${COT_BUDGET_END_TOKEN:-</${REASONING_TAG}>}"
    COT_BUDGET_MAX_TOKENS="${COT_BUDGET_MAX_TOKENS:-128}"
    COT_FORMAT_REWARD_ENABLED="${COT_FORMAT_REWARD_ENABLED:-true}"
    COT_FORMAT_REWARD_MISSING="${COT_FORMAT_REWARD_MISSING:-0.0}"
    ENABLE_RESPONSE_LOSS_WEIGHT_MASK="${ENABLE_RESPONSE_LOSS_WEIGHT_MASK:-false}"
    THOUGHT_LOSS_WEIGHT="${THOUGHT_LOSS_WEIGHT:-0.2}"
    ANSWER_LOSS_WEIGHT="${ANSWER_LOSS_WEIGHT:-1.0}"
    DEFAULT_LOSS_WEIGHT="${DEFAULT_LOSS_WEIGHT:-1.0}"
    ANSWER_FALLBACK_AFTER_THOUGHT="${ANSWER_FALLBACK_AFTER_THOUGHT:-true}"
    VLLM_USE_V1="${VLLM_USE_V1:-1}"
else
    COT_BUDGET_ENABLED="${COT_BUDGET_ENABLED:-false}"
    COT_BUDGET_START_TOKEN="${COT_BUDGET_START_TOKEN:-<think>}"
    COT_BUDGET_END_TOKEN="${COT_BUDGET_END_TOKEN:-</think>}"
    COT_BUDGET_MAX_TOKENS="${COT_BUDGET_MAX_TOKENS:-0}"
fi

export MULTI_TASK_DATA_ROOT TRAIN_FILE TEST_FILE
export TASKS N_GPUS_PER_NODE TP_SIZE ROLLOUT_BS GLOBAL_BS VAL_BATCH_SIZE ROLLOUT_N
export ROLLOUT_TEMPERATURE LR KL_COEF ENTROPY_COEFF
export VAL_BEFORE_TRAIN VAL_FREQ SAVE_FREQ SAVE_BEST SAVE_LIMIT
export ENABLE_GPU_FILLER POST_TRAIN_OCCUPANCY CHECK_EXPERIMENT_JSONL CHECK_EXPERIMENT_FRAME_FILES MIX_FORCE
export REUSE_EXISTING_DATA
export FRAME_SAMPLE_POLICY FRAME_SAMPLE_MAX_FRAMES FRAME_SAMPLE_POLICY_VERSION MAX_FRAMES
export ROLLOUT_MAX_NUM_SEQS ROLLOUT_MAX_BATCHED_TOKENS MAX_RESPONSE_LEN
export COT_BUDGET_ENABLED COT_BUDGET_START_TOKEN COT_BUDGET_END_TOKEN COT_BUDGET_MAX_TOKENS
export ENABLE_RESPONSE_LOSS_WEIGHT_MASK THOUGHT_LOSS_WEIGHT ANSWER_LOSS_WEIGHT DEFAULT_LOSS_WEIGHT ANSWER_FALLBACK_AFTER_THOUGHT
export VLLM_USE_V1

source "${REPO_ROOT}/video_proxy/training/recipes/teacher_train_ema_grpo.sh"

ROLLOUT_JSONL="${CHECKPOINT_ROOT}/${EXP_NAME}/rollouts/step_000001.jsonl"
if [[ "${COT_MODE,,}" =~ ^(true|1|yes)$ ]]; then
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
        echo "[single-teacher] WARN: rollout file not found after smoke run: ${ROLLOUT_JSONL}" >&2
    fi
fi
