#!/usr/bin/env bash
# =============================================================
# exp1_sort.sh — Event Shuffle (Sort) 实验
#
# 用法:
#   bash local_scripts/event_logic_ablations/exp1_sort.sh
#
#   # 自定义参数
#   MAX_STEPS=80 SORT_SEQ_LEN=4 bash local_scripts/event_logic_ablations/exp1_sort.sh
# =============================================================
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-el_sort_exp1}"
DATA_DIR="${DATA_DIR:-${EL_DATA_ROOT}/sort_exp1}"
mkdir -p "${DATA_DIR}"

# Sort 参数
SORT_SEQ_LEN="${SORT_SEQ_LEN:-5}"
MIN_EVENTS="${MIN_EVENTS:-3}"
MAX_EVENTS="${MAX_EVENTS:-8}"
SAMPLES_PER_GROUP="${SAMPLES_PER_GROUP:-1}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_sort_train.sh"
