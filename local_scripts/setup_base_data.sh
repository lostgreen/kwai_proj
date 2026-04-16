#!/usr/bin/env bash
# =============================================================
# setup_base_data.sh — 一键将预构建数据采样到 base/ + val/
#
# 前置条件 (在服务器上先运行):
#   1. TG 训练数据:
#      bash proxy_data/temporal_grounding/run_pipeline.sh
#      → 输出: proxy_data/temporal_grounding/data/tg_train_max256s_validated.jsonl
#
#   2. MCQ: proxy_data/llava_video_178k/ pipeline → results/train_final.jsonl
#   3. Hier Seg: annotation pipeline → train_all.jsonl / val_all.jsonl
#   4. Event Logic (可选): event_logic pipeline → train.jsonl / val.jsonl
#
# TVGBench val 会由本脚本自动从 annotation 构建并采样。
#
# 本脚本只做 copy + sample，不做数据生成。
# 只需运行一次，后续实验复用 base/ 和 val/。
#
# 用法:
#   bash local_scripts/setup_base_data.sh
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# ---- Source common 获取默认路径 ----
source "${SCRIPT_DIR}/multi_task_common.sh"

# ---- 数据源路径 ----
TG_TRAIN_SOURCE="${TG_TRAIN_SOURCE:-${REPO_ROOT}/proxy_data/temporal_grounding/data/tg_train_max256s_validated.jsonl}"
TVGBENCH_JSON="${TVGBENCH_JSON:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset/annotations/tvgbench.json}"
VIDEO_BASE="${VIDEO_BASE:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset}"
MCQ_SOURCE="${MCQ_SOURCE:-${REPO_ROOT}/proxy_data/llava_video_178k/results/train_final.jsonl}"

# ---- 构建参数 ----
_FORCE_FLAG=""
if [[ "${FORCE:-false}" == "true" ]]; then
    _FORCE_FLAG="--force"
fi

echo "============================================"
echo "  Setup Multi-Task Base Data"
echo "============================================"
echo "  Data root:    ${THREE_TASK_DATA_ROOT}"
echo "  Tasks:        ${TASKS}"
echo "  Force:        ${FORCE:-false}"
echo "============================================"

# shellcheck disable=SC2086
python3 -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from local_scripts.data.mixer import main; main()
" -- \
    --data-root "${THREE_TASK_DATA_ROOT}" \
    --tasks ${TASKS} \
    ${_FORCE_FLAG} \
    setup \
    --tg-train-source "${TG_TRAIN_SOURCE}" \
    --tg-tvgbench-json "${TVGBENCH_JSON}" \
    --tg-video-base "${VIDEO_BASE}" \
    --mcq-source "${MCQ_SOURCE}" \
    --hier-val-source "${HIER_VAL_SOURCE}" \
    --val-tg-n "${VAL_TG_N:-150}" \
    --val-mcq-n "${VAL_MCQ_N:-150}" \
    --val-hier-n "${VAL_HIER_N:-150}" \
    ${EL_TRAIN:+--el-train "${EL_TRAIN}"} \
    ${EL_VAL_SOURCE:+--el-val-source "${EL_VAL_SOURCE}"} \
    --val-el-n "${VAL_EL_N:-100}"
