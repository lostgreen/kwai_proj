#!/usr/bin/env bash
# =============================================================
# 实验 4 — Add + Replace
#   MCQ 双任务（add + replace），不含 sort
#   验证两种 MCQ 任务联合训练效果
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-el_ablation_exp4_add_replace}"
DATA_DIR="${DATA_DIR:-${EL_DATA_ROOT}/exp4}"
mkdir -p "${DATA_DIR}"

ADD_PER_VIDEO=2
REPLACE_PER_VIDEO=2
SORT_PER_VIDEO=0

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
