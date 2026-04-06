#!/usr/bin/env bash
# =============================================================
# exp1_sort.sh — L2 Event Sort（phase → events 排序）
#
# 数据需提前构建好（见 build_event_shuffle.py 顶部命令）。
#
# 用法:
#   bash local_scripts/event_logic_ablations/exp1_sort.sh
# =============================================================
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-el_sort_l2_exp1}"
DATA_DIR="${DATA_DIR:-${EL_DATA_ROOT}/sort_l2_exp1}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_sort_train.sh"
