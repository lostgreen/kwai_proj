#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

USER_ROLLOUT_BS="${ROLLOUT_BS:-}"
USER_GLOBAL_BS="${GLOBAL_BS:-}"
USER_VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-}"
USER_MAX_FRAMES="${MAX_FRAMES:-}"
USER_MAX_PIXELS="${MAX_PIXELS:-}"
USER_LR="${LR:-}"
USER_KL_COEF="${KL_COEF:-}"
USER_ENTROPY_COEFF="${ENTROPY_COEFF:-}"

source "${SCRIPT_DIR}/../common/multi_task_common.sh"

MODEL_FAMILY="${MODEL_FAMILY:?set MODEL_FAMILY before sourcing teacher_train_ema_grpo.sh}"
MODEL_SIZE="${MODEL_SIZE:?set MODEL_SIZE before sourcing teacher_train_ema_grpo.sh}"

PROJECT_NAME="${PROJECT_NAME:-VideoProxy-teacher}"
EXP_NAME="${EXP_NAME:-${MODEL_FAMILY}_${MODEL_SIZE}_teacher_ema_grpo}"
TASKS="${TASKS:-tg mcq hier_seg}"

TRAINING_MODE="rl"
ADV_ESTIMATOR="ema_grpo"
ONLINE_FILTERING=false
USE_KL_LOSS=true
DISABLE_KL=false
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
LR="${USER_LR:-1e-6}"
KL_COEF="${USER_KL_COEF:-0.001}"
ENTROPY_COEFF="${USER_ENTROPY_COEFF:-0.0}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"
MAX_FRAMES="${USER_MAX_FRAMES:-48}"
MAX_PIXELS="${USER_MAX_PIXELS:-65536}"

if [[ -n "${USER_ROLLOUT_BS}" ]]; then
    ROLLOUT_BS="${USER_ROLLOUT_BS}"
elif [[ "${MODEL_SIZE}" == "8b" || "${MODEL_SIZE}" == "7b" ]]; then
    ROLLOUT_BS=32
else
    ROLLOUT_BS=64
fi

if [[ -n "${USER_GLOBAL_BS}" ]]; then
    GLOBAL_BS="${USER_GLOBAL_BS}"
else
    GLOBAL_BS="${ROLLOUT_BS}"
fi

if [[ -n "${USER_VAL_BATCH_SIZE}" ]]; then
    VAL_BATCH_SIZE="${USER_VAL_BATCH_SIZE}"
else
    VAL_BATCH_SIZE="${GLOBAL_BS}"
fi

echo "[teacher-ema-grpo] MODEL_FAMILY=${MODEL_FAMILY} MODEL_SIZE=${MODEL_SIZE}"
echo "[teacher-ema-grpo] MODEL_PATH=${MODEL_PATH}"
echo "[teacher-ema-grpo] EXP_NAME=${EXP_NAME} TASKS=${TASKS}"
echo "[teacher-ema-grpo] ROLLOUT_BS=${ROLLOUT_BS} GLOBAL_BS=${GLOBAL_BS} N=${ROLLOUT_N}"
echo "[teacher-ema-grpo] LR=${LR} KL_COEF=${KL_COEF} MAX_FRAMES=${MAX_FRAMES} MAX_PIXELS=${MAX_PIXELS}"

source "${SCRIPT_DIR}/../launchers/run_multi_task.sh"
