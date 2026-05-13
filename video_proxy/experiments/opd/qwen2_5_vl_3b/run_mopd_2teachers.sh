#!/usr/bin/env bash
# Qwen2.5-VL-3B base+R1/R2 MOPD with AOT and SEG teachers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FAMILY="qwen2_5_vl"
MODEL_SIZE="3b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen2.5-VL-3B-Instruct}"
EXP_NAME="${EXP_NAME:-mopd_qwen2_5vl3b_base_r1_r2_2teachers_bs64_mf256_epoch1_save50_keep1}"
PROJECT_NAME="${PROJECT_NAME:-VideoProxy-opd-comparison-3b}"
SAVE_LIMIT="${SAVE_LIMIT:-1}"
SAVE_BEST="${SAVE_BEST:-true}"
TP_SIZE="${TP_SIZE:-1}"

source "${SCRIPT_DIR}/../common_mopd.sh"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${MOPD_CHECKPOINT_ROOT_3B}}"
mopd_base_r1_r2_data_defaults
mopd_8gpu_defaults
mopd_full_epoch_save_defaults
mopd_training_defaults
mopd_two_teacher_defaults
mopd_validate_rollout_tokens
validate_mopd_teacher_paths

source "${SCRIPT_DIR}/../../../training/recipes/opd_train.sh"
