#!/usr/bin/env bash
# 4B full-composition MOPD with three 4B teachers.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PROJECT_NAME="${PROJECT_NAME:-VideoProxy-opd-comparison-4b}"
opd_comparison_full_data_defaults
opd_comparison_8gpu_defaults
opd_comparison_full_epoch_save_defaults
opd_comparison_mopd_defaults

MODEL_PATH="${MODEL_PATH:-${QWEN3_VL_4B_MODEL_PATH}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${CHECKPOINT_ROOT_4B_COMPARISON}}"
EXP_NAME="${EXP_NAME:-mopd_qwen3vl4b_full_comp_4b_teachers_bs64_mf256_epoch1_save50}"
TP_SIZE="${TP_SIZE:-1}"

opd_comparison_validate_rollout_tokens

source "${REPO_ROOT_LOCAL}/local_scripts/run_multi_task.sh"
