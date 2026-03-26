#!/usr/bin/env bash
# =============================================================
# 实验 1 — Add Only
#   仅 add（预测下一步）任务，消融单任务效果
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-el_ablation_exp1_add_only}"
DATA_DIR="${DATA_DIR:-${EL_DATA_ROOT}/exp1}"
mkdir -p "${DATA_DIR}"

ADD_PER_VIDEO=3
REPLACE_PER_VIDEO=0
SORT_PER_VIDEO=0

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
