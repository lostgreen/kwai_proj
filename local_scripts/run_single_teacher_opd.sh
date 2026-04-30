#!/usr/bin/env bash
# ============================================================
# Single-teacher OPD smoke launcher.
#
# This uses student on-policy rollout (n=1) plus teacher top-k sparse KL.
# It does not compute reward groups or GRPO advantages.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${TEACHER_MODEL_PATH:-}" ]]; then
    echo "[single-teacher-opd] ERROR: set TEACHER_MODEL_PATH=/path/to/teacher" >&2
    exit 1
fi

EXP_NAME="${EXP_NAME:-single_teacher_opd_2gpu_smoke}"
TRAINING_MODE="opd"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
DISABLE_KL=false
USE_KL_LOSS=false
ONLINE_FILTERING=false
ROLLOUT_N=1
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
OPD_TOPK="${OPD_TOPK:-20}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
NNODES="${NNODES:-1}"
TP_SIZE="${TP_SIZE:-2}"
ROLLOUT_BS="${ROLLOUT_BS:-8}"
GLOBAL_BS="${GLOBAL_BS:-8}"
MB_PER_UPDATE="${MB_PER_UPDATE:-1}"
MB_PER_EXP="${MB_PER_EXP:-1}"
MAX_FRAMES="${MAX_FRAMES:-128}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.35}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-8192}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
VAL_FREQ="${VAL_FREQ:--1}"

source "${SCRIPT_DIR}/run_multi_task.sh"
