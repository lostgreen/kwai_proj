#!/usr/bin/env bash
# ============================================================
# Multi-teacher OPD smoke launcher for 2 GPUs.
#
# Student rollout uses n=1. Teacher top-k sparse KL is routed by
# problem_type:
#   - seg_aot_*         -> aot teacher
#   - temporal_seg_*    -> seg teacher
#   - event_logic_*     -> eventlogic teacher
#
# Actor/ref params are offloaded by default so only the active teacher is
# loaded to GPU for its sub-batch.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

AOT_TEACHER_MODEL_PATH="${AOT_TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations/composition_base_aot_aot10k_mf256_ema/global_step_200/actor/huggingface}"
SEG_TEACHER_MODEL_PATH="${SEG_TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations/composition_base_seg_hier10k_mf256_ema/global_step_250/actor/huggingface}"
EVENTLOGIC_TEACHER_MODEL_PATH="${EVENTLOGIC_TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations/composition_base_aot_logic_aot10k_el10k_mf256_ema/global_step_300/actor/huggingface}"

for _teacher_var in AOT_TEACHER_MODEL_PATH SEG_TEACHER_MODEL_PATH EVENTLOGIC_TEACHER_MODEL_PATH; do
    if [[ -z "${!_teacher_var:-}" ]]; then
        echo "[multi-teacher-opd] ERROR: set ${_teacher_var}=/path/to/teacher" >&2
        exit 1
    fi
done

EXP_NAME="${EXP_NAME:-multi_teacher_opd_2gpu_smoke}"
TASKS="${TASKS:-hier_seg aot event_logic}"
TASK_HOMOGENEOUS_BATCHING="${TASK_HOMOGENEOUS_BATCHING:-true}"
TRAINING_MODE="opd"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
DISABLE_KL=false
USE_KL_LOSS=false
ONLINE_FILTERING=false
ROLLOUT_N=1
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
OPD_TOPK="${OPD_TOPK:-20}"
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
OPD_TEACHER_KEY="${OPD_TEACHER_KEY:-problem_type}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
NNODES="${NNODES:-1}"
TP_SIZE="${TP_SIZE:-2}"
ROLLOUT_BS="${ROLLOUT_BS:-8}"
GLOBAL_BS="${GLOBAL_BS:-8}"
MB_PER_UPDATE="${MB_PER_UPDATE:-1}"
MB_PER_EXP="${MB_PER_EXP:-1}"
MAX_FRAMES="${MAX_FRAMES:-128}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-14000}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.35}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-20480}"
ACTOR_OFFLOAD_PARAMS="${ACTOR_OFFLOAD_PARAMS:-true}"
ACTOR_OFFLOAD_OPTIMIZER="${ACTOR_OFFLOAD_OPTIMIZER:-true}"
REF_OFFLOAD_PARAMS="${REF_OFFLOAD_PARAMS:-true}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
VAL_FREQ="${VAL_FREQ:--1}"

MIN_ROLLOUT_MAX_BATCHED_TOKENS=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
if (( ROLLOUT_MAX_BATCHED_TOKENS < MIN_ROLLOUT_MAX_BATCHED_TOKENS )); then
    echo "[multi-teacher-opd] ERROR: ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS} must be >= MAX_PROMPT_LEN + MAX_RESPONSE_LEN = ${MIN_ROLLOUT_MAX_BATCHED_TOKENS}" >&2
    exit 1
fi

source "${SCRIPT_DIR}/run_multi_task.sh"
