#!/usr/bin/env bash
# Shared MOPD settings restored from the pre-layout-refactor OPD comparison launchers.

MOPD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_MULTI_TASK_DATA_ROOT="/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task"
MULTI_TASK_DATA_ROOT="${MULTI_TASK_DATA_ROOT:-${THREE_TASK_DATA_ROOT:-${DEFAULT_MULTI_TASK_DATA_ROOT}}}"

FULL_COMPOSITION_EXP_NAME="${FULL_COMPOSITION_EXP_NAME:-composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema}"
FULL_COMPOSITION_DATA_DIR="${FULL_COMPOSITION_DATA_DIR:-${MULTI_TASK_DATA_ROOT}/experiments/${FULL_COMPOSITION_EXP_NAME}}"
FULL_COMPOSITION_TRAIN_FILE="${FULL_COMPOSITION_TRAIN_FILE:-${FULL_COMPOSITION_DATA_DIR}/train.jsonl}"
FULL_COMPOSITION_TEST_FILE="${FULL_COMPOSITION_TEST_FILE:-${FULL_COMPOSITION_DATA_DIR}/val.jsonl}"

BASE_R1_R2_COMPOSITION_EXP_NAME="${BASE_R1_R2_COMPOSITION_EXP_NAME:-composition_base_seg_aot_hier10k_aot10k_mf256_ema}"
BASE_R1_R2_COMPOSITION_DATA_DIR="${BASE_R1_R2_COMPOSITION_DATA_DIR:-${MULTI_TASK_DATA_ROOT}/experiments/${BASE_R1_R2_COMPOSITION_EXP_NAME}}"
BASE_R1_R2_COMPOSITION_TRAIN_FILE="${BASE_R1_R2_COMPOSITION_TRAIN_FILE:-${BASE_R1_R2_COMPOSITION_DATA_DIR}/train.jsonl}"
BASE_R1_R2_COMPOSITION_TEST_FILE="${BASE_R1_R2_COMPOSITION_TEST_FILE:-${BASE_R1_R2_COMPOSITION_DATA_DIR}/val.jsonl}"

MOPD_TEACHER_CKPT_ROOT="${MOPD_TEACHER_CKPT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations}"
MOPD_CHECKPOINT_ROOT_3B="${MOPD_CHECKPOINT_ROOT_3B:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/opd_comparison_3b}"
MOPD_CHECKPOINT_ROOT_4B="${MOPD_CHECKPOINT_ROOT_4B:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/opd_comparison_4b}"
MOPD_CHECKPOINT_ROOT_7B="${MOPD_CHECKPOINT_ROOT_7B:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/opd_comparison_7b}"
MOPD_CHECKPOINT_ROOT_8B="${MOPD_CHECKPOINT_ROOT_8B:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/opd_comparison_8b}"

AOT_TEACHER_STEP="${AOT_TEACHER_STEP:-200}"
SEG_TEACHER_STEP="${SEG_TEACHER_STEP:-250}"
EVENTLOGIC_TEACHER_STEP="${EVENTLOGIC_TEACHER_STEP:-272}"

validate_mopd_teacher_paths() {
    local missing=0
    local teacher_name teacher_prefix teacher_path_var teacher_path
    read -r -a _mopd_teacher_set <<< "${OPD_TEACHER_SET}"
    for teacher_name in "${_mopd_teacher_set[@]}"; do
        teacher_prefix="$(printf '%s' "${teacher_name}" | tr '[:lower:]' '[:upper:]')"
        teacher_path_var="${teacher_prefix}_TEACHER_MODEL_PATH"
        teacher_path="${!teacher_path_var:-}"
        if [[ -z "${teacher_path}" ]]; then
            echo "[mopd] ERROR: ${teacher_path_var} is empty" >&2
            missing=1
        elif [[ ! -f "${teacher_path}/config.json" ]]; then
            echo "[mopd] ERROR: ${teacher_path_var} does not contain config.json: ${teacher_path}" >&2
            missing=1
        fi
    done

    if (( missing != 0 )); then
        echo "[mopd] Set *_TEACHER_MODEL_PATH or *_TEACHER_STEP to existing merged HuggingFace checkpoints." >&2
        exit 1
    fi
}

mopd_full_composition_data_defaults() {
    TRAIN_FILE="${TRAIN_FILE:-${FULL_COMPOSITION_TRAIN_FILE}}"
    TEST_FILE="${TEST_FILE:-${FULL_COMPOSITION_TEST_FILE}}"
    TASKS="${TASKS:-tg mcq hier_seg event_logic aot}"
    HIER_TARGET="${HIER_TARGET:-10000}"
    EL_HARDER_DATA="${EL_HARDER_DATA:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/event_logic_harder}"
    EL_TRAIN="${EL_TRAIN:-${EL_HARDER_DATA}/train_10k.jsonl}"
    EL_VAL_SOURCE="${EL_VAL_SOURCE:-${EL_HARDER_DATA}/val_logic.jsonl}"
    EL_TARGET="${EL_TARGET:-10000}"
    VAL_EL_N="${VAL_EL_N:-300}"
    AOT_TARGET="${AOT_TARGET:-10000}"
    VAL_AOT_N="${VAL_AOT_N:-300}"
}

mopd_base_r1_r2_data_defaults() {
    TRAIN_FILE="${TRAIN_FILE:-${BASE_R1_R2_COMPOSITION_TRAIN_FILE}}"
    TEST_FILE="${TEST_FILE:-${BASE_R1_R2_COMPOSITION_TEST_FILE}}"
    TASKS="${TASKS:-tg mcq hier_seg aot}"
    HIER_TARGET="${HIER_TARGET:-10000}"
    AOT_TARGET="${AOT_TARGET:-10000}"
    VAL_AOT_N="${VAL_AOT_N:-300}"
}

mopd_8gpu_defaults() {
    N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
    NNODES="${NNODES:-1}"
    MB_PER_UPDATE="${MB_PER_UPDATE:-1}"
    MB_PER_EXP="${MB_PER_EXP:-1}"
    MAX_FRAMES="${MAX_FRAMES:-256}"
    MAX_PIXELS="${MAX_PIXELS:-65536}"
    MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-14000}"
    MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}"
    ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"
    ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-20480}"
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
    DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
    DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
    DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-true}"
}

mopd_full_epoch_save_defaults() {
    TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
    if [[ ! "${ALLOW_MAX_STEPS_OVERRIDE:-false}" =~ ^(true|1|yes)$ ]]; then
        MAX_STEPS=""
    else
        MAX_STEPS="${MAX_STEPS:-}"
    fi
    SAVE_FREQ="${SAVE_FREQ:-50}"
    SAVE_LIMIT="${SAVE_LIMIT:--1}"
    VAL_FREQ="${VAL_FREQ:-50}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
}

mopd_training_defaults() {
    TRAINING_MODE="opd"
    ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
    DISABLE_KL=false
    USE_KL_LOSS=false
    ONLINE_FILTERING=false
    ROLLOUT_N=1
    ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
    LR="${LR:-5e-7}"
    KL_COEF="${KL_COEF:-0.01}"
    ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
    OPD_TOPK="${OPD_TOPK:-10}"
    OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
    OPD_TEACHER_KEY="${OPD_TEACHER_KEY:-problem_type}"
    TASK_HOMOGENEOUS_BATCHING="${TASK_HOMOGENEOUS_BATCHING:-true}"
    TASK_HOMOGENEOUS_GROUPING="${TASK_HOMOGENEOUS_GROUPING:-opd_task_group}"
    ACTOR_OFFLOAD_PARAMS="${ACTOR_OFFLOAD_PARAMS:-true}"
    ACTOR_OFFLOAD_OPTIMIZER="${ACTOR_OFFLOAD_OPTIMIZER:-true}"
    REF_OFFLOAD_PARAMS="${REF_OFFLOAD_PARAMS:-true}"
    ROLLOUT_BS="${ROLLOUT_BS:-64}"
    GLOBAL_BS="${GLOBAL_BS:-64}"
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
    ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
}

mopd_three_teacher_defaults() {
    OPD_TEACHER_SET="${OPD_TEACHER_SET:-aot seg eventlogic}"
    AOT_TEACHER_MODEL_PATH="${AOT_TEACHER_MODEL_PATH:-${MOPD_TEACHER_CKPT_ROOT}/composition_base_aot_aot10k_mf256_ema/global_step_${AOT_TEACHER_STEP}/actor/huggingface}"
    SEG_TEACHER_MODEL_PATH="${SEG_TEACHER_MODEL_PATH:-${MOPD_TEACHER_CKPT_ROOT}/composition_base_seg_hier10k_mf256_ema/global_step_${SEG_TEACHER_STEP}/actor/huggingface}"
    EVENTLOGIC_TEACHER_MODEL_PATH="${EVENTLOGIC_TEACHER_MODEL_PATH:-${MOPD_TEACHER_CKPT_ROOT}/composition_base_logic_el10k_mf256_ema/global_step_${EVENTLOGIC_TEACHER_STEP}/actor/huggingface}"
}

mopd_two_teacher_defaults() {
    OPD_TEACHER_SET="${OPD_TEACHER_SET:-aot seg}"
    AOT_TEACHER_MODEL_PATH="${AOT_TEACHER_MODEL_PATH:-${MOPD_TEACHER_CKPT_ROOT}/composition_base_aot_aot10k_mf256_ema/global_step_${AOT_TEACHER_STEP}/actor/huggingface}"
    SEG_TEACHER_MODEL_PATH="${SEG_TEACHER_MODEL_PATH:-${MOPD_TEACHER_CKPT_ROOT}/composition_base_seg_hier10k_mf256_ema/global_step_${SEG_TEACHER_STEP}/actor/huggingface}"
    EVENTLOGIC_TEACHER_MODEL_PATH=""
}

mopd_validate_rollout_tokens() {
    local min_rollout_max_batched_tokens=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))
    if (( ROLLOUT_MAX_BATCHED_TOKENS < min_rollout_max_batched_tokens )); then
        echo "[mopd] ERROR: ROLLOUT_MAX_BATCHED_TOKENS=${ROLLOUT_MAX_BATCHED_TOKENS} must be >= MAX_PROMPT_LEN + MAX_RESPONSE_LEN = ${min_rollout_max_batched_tokens}" >&2
        exit 1
    fi
}
