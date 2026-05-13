#!/usr/bin/env bash
# Qwen3-VL-4B full-composition MOPD with task-specific AOT/SEG/event-logic teachers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FAMILY="qwen3_vl"
MODEL_SIZE="4b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
EXP_NAME="${EXP_NAME:-mopd_qwen3vl4b_full_comp_4b_teachers_bs64_mf256_epoch1_save50}"
PROJECT_NAME="${PROJECT_NAME:-VideoProxy-opd-comparison-4b}"
TP_SIZE="${TP_SIZE:-1}"

source "${SCRIPT_DIR}/../common_mopd.sh"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${MOPD_CHECKPOINT_ROOT_4B}}"
mopd_full_composition_data_defaults
mopd_8gpu_defaults
mopd_full_epoch_save_defaults
mopd_training_defaults
mopd_three_teacher_defaults
mopd_validate_rollout_tokens
validate_mopd_teacher_paths

source "${SCRIPT_DIR}/../../../training/recipes/opd_train.sh"
