#!/usr/bin/env bash
# Qwen3-VL-8B full-composition MOPD with task-specific AOT/SEG/event-logic teachers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FAMILY="qwen3_vl"
MODEL_SIZE="8b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct}"
EXP_NAME="${EXP_NAME:-mopd_qwen3vl8b_full_comp_4b_teachers_bs64_mf256_epoch1_save50_keep1}"
PROJECT_NAME="${PROJECT_NAME:-VideoProxy-opd-comparison-8b}"
SAVE_LIMIT="${SAVE_LIMIT:-1}"
SAVE_BEST="${SAVE_BEST:-true}"
TP_SIZE="${TP_SIZE:-2}"
ENABLE_GPU_FILLER="${ENABLE_GPU_FILLER:-false}"

source "${SCRIPT_DIR}/../common_mopd.sh"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${MOPD_CHECKPOINT_ROOT_8B}}"
mopd_full_composition_data_defaults
mopd_8gpu_defaults
mopd_full_epoch_save_defaults
mopd_training_defaults
mopd_three_teacher_defaults
mopd_validate_rollout_tokens
validate_mopd_teacher_paths

source "${SCRIPT_DIR}/../../../training/recipes/opd_train.sh"
