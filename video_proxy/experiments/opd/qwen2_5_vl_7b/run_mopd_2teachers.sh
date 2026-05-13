#!/usr/bin/env bash
# Qwen2.5-VL-7B base+R1/R2 MOPD with AOT and SEG teachers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FAMILY="qwen2_5_vl"
MODEL_SIZE="7b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen2.5-VL-7B-Instruct}"
EXP_NAME="${EXP_NAME:-mopd_qwen2_5vl7b_base_r1_r2_2teachers_bs64_mf256_epoch1_save50_keep1}"
PROJECT_NAME="${PROJECT_NAME:-VideoProxy-opd-comparison-7b}"
SAVE_LIMIT="${SAVE_LIMIT:-1}"
SAVE_BEST="${SAVE_BEST:-true}"
TP_SIZE="${TP_SIZE:-2}"

source "${SCRIPT_DIR}/../common_mopd.sh"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${MOPD_CHECKPOINT_ROOT_7B}}"
AOT_TEACHER_MODEL_PATH="${AOT_TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen2_5_vl_7b_aot_teacher_nocot/global_step_250/actor/huggingface}"
SEG_TEACHER_MODEL_PATH="${SEG_TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen2_5_vl_7b_seg_teacher_nocot/global_step_272/actor/huggingface}"
mopd_base_r1_r2_data_defaults
mopd_8gpu_defaults
mopd_full_epoch_save_defaults
mopd_training_defaults
mopd_two_teacher_defaults
mopd_validate_rollout_tokens
validate_mopd_teacher_paths

source "${SCRIPT_DIR}/../../../training/recipes/opd_train.sh"
