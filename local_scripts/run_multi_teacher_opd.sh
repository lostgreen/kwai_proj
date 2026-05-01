#!/usr/bin/env bash
# ============================================================
# Multi-teacher OPD smoke launcher for 2 GPUs.
#
# Student rollout uses n=1. Teacher top-k sparse KL is routed by
# problem_type:
#   - temporal_grounding -> aot teacher
#   - llava_mcq          -> aot teacher
#   - seg_aot_*         -> aot teacher
#   - temporal_seg_*    -> seg teacher
#   - event_logic_*     -> eventlogic teacher
#
# Actor/ref params are offloaded by default so only the active teacher is
# loaded to GPU for its sub-batch.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
AOT_TEACHER_MODEL_PATH="${AOT_TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations/composition_base_aot_aot10k_mf256_ema/global_step_200/actor/huggingface}"
SEG_TEACHER_MODEL_PATH="${SEG_TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations/composition_base_seg_hier10k_mf256_ema/global_step_250/actor/huggingface}"
EVENTLOGIC_TEACHER_MODEL_PATH="${EVENTLOGIC_TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations/composition_base_logic_el10k_mf256_ema/global_step_272/actor/huggingface}"

for _teacher_var in AOT_TEACHER_MODEL_PATH SEG_TEACHER_MODEL_PATH EVENTLOGIC_TEACHER_MODEL_PATH; do
    if [[ -z "${!_teacher_var:-}" ]]; then
        echo "[multi-teacher-opd] ERROR: set ${_teacher_var}=/path/to/teacher" >&2
        exit 1
    fi
done

validate_opd_teacher_paths() {
    local missing=0
    local teacher_name teacher_path_var teacher_path
    for teacher_name in AOT SEG EVENTLOGIC; do
        teacher_path_var="${teacher_name}_TEACHER_MODEL_PATH"
        teacher_path="${!teacher_path_var:-}"
        if [[ -z "${teacher_path}" ]]; then
            echo "[multi-teacher-opd] ERROR: ${teacher_path_var} is empty" >&2
            missing=1
        elif [[ ! -f "${teacher_path}/config.json" ]]; then
            echo "[multi-teacher-opd] ERROR: ${teacher_path_var} does not contain config.json: ${teacher_path}" >&2
            missing=1
        fi
    done

    if (( missing != 0 )); then
        echo "[multi-teacher-opd] Set *_TEACHER_MODEL_PATH to an existing merged HuggingFace checkpoint." >&2
        exit 1
    fi
}

EXP_NAME="${EXP_NAME:-multi_teacher_opd_2gpu_mf256_sanity}"
TRAIN_FILE="${TRAIN_FILE:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/experiments/composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema/train.jsonl}"
TEST_FILE="${TEST_FILE:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/experiments/composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema/val.jsonl}"
TASKS="${TASKS:-tg mcq hier_seg event_logic aot}"
HIER_TARGET="${HIER_TARGET:-10000}"
EL_HARDER_DATA="${EL_HARDER_DATA:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/event_logic_harder}"
EL_TRAIN="${EL_TRAIN:-${EL_HARDER_DATA}/train_10k.jsonl}"
EL_VAL_SOURCE="${EL_VAL_SOURCE:-${EL_HARDER_DATA}/val_logic.jsonl}"
EL_TARGET="${EL_TARGET:-10000}"
VAL_EL_N="${VAL_EL_N:-300}"
AOT_TARGET="${AOT_TARGET:-10000}"
VAL_AOT_N="${VAL_AOT_N:-300}"
TASK_HOMOGENEOUS_BATCHING="${TASK_HOMOGENEOUS_BATCHING:-true}"
TASK_HOMOGENEOUS_GROUPING="${TASK_HOMOGENEOUS_GROUPING:-opd_task_group}"
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
OPD_TEACHER_KEY="${OPD_TEACHER_KEY:-problem_type}"

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
ACTOR_OFFLOAD_PARAMS="${ACTOR_OFFLOAD_PARAMS:-true}"
ACTOR_OFFLOAD_OPTIMIZER="${ACTOR_OFFLOAD_OPTIMIZER:-true}"
REF_OFFLOAD_PARAMS="${REF_OFFLOAD_PARAMS:-true}"
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
    echo "[multi-teacher-opd] ERROR: ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS} must be >= MAX_PROMPT_LEN + MAX_RESPONSE_LEN = ${MIN_ROLLOUT_MAX_BATCHED_TOKENS}" >&2
    exit 1
fi
validate_opd_teacher_paths

source "${SCRIPT_DIR}/run_multi_task.sh"
