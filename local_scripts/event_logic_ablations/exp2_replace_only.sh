#!/usr/bin/env bash
# =============================================================
# 实验 2 — Replace Only
#   仅 replace（缺失补全）任务，消融单任务效果
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-el_ablation_exp2_replace_only}"
DATA_DIR="${DATA_DIR:-${EL_DATA_ROOT}/exp2}"
mkdir -p "${DATA_DIR}"

ADD_PER_VIDEO=0
REPLACE_PER_VIDEO=3
SORT_PER_VIDEO=0

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
