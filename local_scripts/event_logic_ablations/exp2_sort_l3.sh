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

# ---- 构建数据 ----
if [[ ! -f "${DATA_DIR}/train.jsonl" || "${FORCE_BUILD:-false}" == "true" ]]; then
  echo "[el] Building L3 action sort data ..."
  python3 "${REPO_ROOT}/proxy_data/youcook2_seg/event_logic/build_event_shuffle.py" \
    --annotation-dir "${ANNOTATION_DIR}" \
    --clip-dir       "${CLIP_DIR}" \
    --output-dir     "${DATA_DIR}" \
    --level          l3 \
    --filter-order \
    --min-events     "${MIN_EVENTS:-3}" \
    --max-events     "${MAX_EVENTS:-8}" \
    --seq-len        "${SORT_SEQ_LEN:-5}" \
    --samples-per-group "${SAMPLES_PER_GROUP:-1}" \
    --complete-only \
    --val-count      "${VAL_COUNT:-100}" \
    --seed           "${BUILD_SEED:-42}"
fi

# ---- 训练 ----
source "$(dirname "${BASH_SOURCE[0]}")/launch_sort_train.sh"
