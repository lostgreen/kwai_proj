#!/usr/bin/env bash
# ============================================================
# Single-teacher OPD smoke launcher.
#
# This uses student on-policy rollout (n=1) plus teacher top-k sparse KL.
# It does not compute reward groups or GRPO advantages.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations/composition_base_aot_aot10k_mf256_ema/global_step_200/actor/huggingface}"

if [[ -z "${TEACHER_MODEL_PATH:-}" ]]; then
    echo "[single-teacher-opd] ERROR: set TEACHER_MODEL_PATH=/path/to/teacher" >&2
    exit 1
fi

EXP_NAME="${EXP_NAME:-opd_qwen3vl4b_from_base_aot_teacher_step200_sanity}"
TRAIN_FILE="${TRAIN_FILE:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/experiments/composition_base_aot_aot10k_mf256_ema/train.jsonl}"
TEST_FILE="${TEST_FILE:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/experiments/composition_base_aot_aot10k_mf256_ema/val.jsonl}"
TASKS="${TASKS:-tg mcq aot}"
TRAINING_MODE="opd"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
DISABLE_KL=false
USE_KL_LOSS=false
ONLINE_FILTERING=false
ROLLOUT_N=1
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
OPD_TOPK="${OPD_TOPK:-10}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
NNODES="${NNODES:-1}"
TP_SIZE="${TP_SIZE:-1}"
ROLLOUT_BS="${ROLLOUT_BS:-16}"
GLOBAL_BS="${GLOBAL_BS:-16}"
MB_PER_UPDATE="${MB_PER_UPDATE:-1}"
MB_PER_EXP="${MB_PER_EXP:-1}"
MAX_FRAMES="${MAX_FRAMES:-256}"
MAX_PIXELS="${MAX_PIXELS:-65536}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-14000}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.35}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-20480}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
VAL_FREQ="${VAL_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
SAVE_LIMIT="${SAVE_LIMIT:-3}"
MAX_STEPS="${MAX_STEPS:-50}"

MIN_ROLLOUT_MAX_BATCHED_TOKENS=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
if (( ROLLOUT_MAX_BATCHED_TOKENS < MIN_ROLLOUT_MAX_BATCHED_TOKENS )); then
    echo "[single-teacher-opd] ERROR: ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS} must be >= MAX_PROMPT_LEN + MAX_RESPONSE_LEN = ${MIN_ROLLOUT_MAX_BATCHED_TOKENS}" >&2
    exit 1
fi

source "${SCRIPT_DIR}/run_multi_task.sh"
