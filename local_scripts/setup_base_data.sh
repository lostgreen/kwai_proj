#!/usr/bin/env bash
# =============================================================
# setup_base_data.sh — 一键将预构建数据采样到 base/ + val/
#
# 前置条件 (在服务器上先运行):
#   1. TG 数据 (TimeRFT + TVGBench 分别处理):
#      TVGBENCH_JSON=/path/to/tvgbench.json bash proxy_data/temporal_grounding/run_pipeline.sh
#      → tg_timerft_max256s_validated.jsonl  (train)
#      → tg_tvgbench_max256s_validated.jsonl (val 采样源)
#
#   2. MCQ: proxy_data/llava_video_178k/ pipeline → results/train_final.jsonl
#   3. Hier Seg: annotation pipeline → train_all.jsonl / val_all.jsonl
#   4. Event Logic (可选): event_logic pipeline → train.jsonl / val.jsonl
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
TG_TRAIN_SOURCE="${TG_TRAIN_SOURCE:-${REPO_ROOT}/proxy_data/temporal_grounding/data/tg_timerft_max256s_validated.jsonl}"
TG_TVGBENCH_SOURCE="${TG_TVGBENCH_SOURCE:-${REPO_ROOT}/proxy_data/temporal_grounding/data/tg_tvgbench_max256s_validated.jsonl}"
MCQ_SOURCE="${MCQ_SOURCE:-${REPO_ROOT}/proxy_data/llava_video_178k/results/train_final_direct.jsonl}"

# ---- 构建参数 ----
_FORCE_FLAG=""
if [[ "${FORCE:-false}" == "true" ]]; then
    _FORCE_FLAG="--force"
fi

echo "============================================"
echo "  Setup Multi-Task Base Data"
echo "============================================"
echo "  Data root:    ${MULTI_TASK_DATA_ROOT}"
echo "  Tasks:        ${TASKS}"
echo "  Force:        ${FORCE:-false}"
echo "  Val TG N:     ${VAL_TG_N:-600}"
echo "  Val MCQ N:    ${VAL_MCQ_N:-600}"
echo "============================================"

# shellcheck disable=SC2086
python3 -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from local_scripts.data.mixer import main; main()
" \
    --data-root "${MULTI_TASK_DATA_ROOT}" \
    ${_FORCE_FLAG} \
    setup \
    --tasks ${TASKS} \
    --tg-train-source "${TG_TRAIN_SOURCE}" \
    --tg-tvgbench-source "${TG_TVGBENCH_SOURCE}" \
    --mcq-source "${MCQ_SOURCE}" \
    --hier-val-source "${HIER_VAL_SOURCE}" \
    --val-tg-n "${VAL_TG_N:-600}" \
    --val-mcq-n "${VAL_MCQ_N:-600}" \
    --val-hier-n "${VAL_HIER_N:-150}" \
    ${EL_TRAIN:+--el-train "${EL_TRAIN}"} \
    ${EL_VAL_SOURCE:+--el-val-source "${EL_VAL_SOURCE}"} \
    --val-el-n "${VAL_EL_N:-100}" \
    ${AOT_TRAIN:+--aot-train "${AOT_TRAIN}"} \
    ${AOT_VAL_SOURCE:+--aot-val-source "${AOT_VAL_SOURCE}"} \
    --val-aot-n "${VAL_AOT_N:-300}"
