#!/usr/bin/env bash
# =============================================================
# exp2_sort_l3.sh — L3 Action Sort（event → actions 排序）
#
# 数据需提前构建好（见 build_event_shuffle.py 顶部命令）。
#
# 用法:
#   bash local_scripts/event_logic_ablations/exp2_sort_l3.sh
# =============================================================
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-el_sort_l3_exp2}"
DATA_DIR="${DATA_DIR:-${EL_DATA_ROOT}/sort_l3_exp2}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_sort_train.sh"
