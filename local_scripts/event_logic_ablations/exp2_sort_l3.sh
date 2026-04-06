#!/usr/bin/env bash
# =============================================================
# exp2_sort_l3.sh — L3 Action Sort（event → actions 排序）
#
# 筛选 _order_distinguishable=true 的 L2 events，shuffle 其 L3 actions。
#
# 用法:
#   bash local_scripts/event_logic_ablations/exp2_sort_l3.sh
# =============================================================
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-el_sort_l3_exp2}"
DATA_DIR="${DATA_DIR:-${EL_DATA_ROOT}/sort_l3_exp2}"
mkdir -p "${DATA_DIR}"

# Sort 参数
SORT_LEVEL="l3"
FILTER_ORDER="true"
SORT_SEQ_LEN="${SORT_SEQ_LEN:-5}"
MIN_EVENTS="${MIN_EVENTS:-3}"
MAX_EVENTS="${MAX_EVENTS:-8}"
SAMPLES_PER_GROUP="${SAMPLES_PER_GROUP:-1}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_sort_train.sh"
