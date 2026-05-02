#!/usr/bin/env bash
# 8B student MOPD with three 4B teachers on full composition data.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PROJECT_NAME="${PROJECT_NAME:-VideoProxy-opd-comparison-8b}"
opd_comparison_full_data_defaults
opd_comparison_8gpu_defaults
SAVE_LIMIT="${SAVE_LIMIT:-1}"
SAVE_BEST="${SAVE_BEST:-true}"
opd_comparison_full_epoch_save_defaults
opd_comparison_mopd_defaults

MODEL_PATH="${MODEL_PATH:-${QWEN3_VL_8B_MODEL_PATH}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${CHECKPOINT_ROOT_8B_COMPARISON}}"
EXP_NAME="${EXP_NAME:-mopd_qwen3vl8b_full_comp_4b_teachers_bs64_mf256_epoch1_save50_keep1}"
TP_SIZE="${TP_SIZE:-2}"
ENABLE_GPU_FILLER="${ENABLE_GPU_FILLER:-false}"

opd_comparison_validate_rollout_tokens
validate_opd_teacher_paths

source "${REPO_ROOT_LOCAL}/local_scripts/run_multi_task.sh"
