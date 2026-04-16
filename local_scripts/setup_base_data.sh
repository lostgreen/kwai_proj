#!/usr/bin/env bash
# =============================================================
# setup_base_data.sh — 一键将预构建数据采样到 base/ + val/
#
# 前置条件:
#   1. TG 数据: 先运行 proxy_data/temporal_grounding/run_pipeline.sh
#   2. MCQ 数据: 先运行 proxy_data/llava_video_178k/ pipeline
#   3. Hier Seg 数据: 先运行 annotation pipeline 得到 train_all/val_all
#   4. Event Logic: 先运行 event_logic pipeline 得到 train/val
#
# 本脚本只做 copy + sample，不做数据生成。
# 只需运行一次，后续实验复用 base/ 和 val/。
#
# 用法:
#   bash local_scripts/setup_base_data.sh
#
# 环境变量 (可选覆盖):
#   THREE_TASK_DATA_ROOT   数据根目录
#   TASKS                  启用的任务列表 (空格分隔)
#   FORCE                  强制重新生成 (true/false)
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# ---- Source common 获取默认路径 ----
source "${SCRIPT_DIR}/multi_task_common.sh"

# ---- 数据源路径 (预构建好的 JSONL) ----
TG_TRAIN_SOURCE="${TG_TRAIN_SOURCE:-${REPO_ROOT}/proxy_data/temporal_grounding/data/tg_train_max256s_validated.jsonl}"
TG_TVGBENCH_SOURCE="${TG_TVGBENCH_SOURCE:-${REPO_ROOT}/proxy_data/temporal_grounding/data/tg_tvgbench_max256s_validated.jsonl}"
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
    --tg-tvgbench-source "${TG_TVGBENCH_SOURCE}" \
    --mcq-source "${MCQ_SOURCE}" \
    --hier-val-source "${HIER_VAL_SOURCE}" \
    --val-tg-n "${VAL_TG_N:-150}" \
    --val-mcq-n "${VAL_MCQ_N:-150}" \
    --val-hier-n "${VAL_HIER_N:-150}" \
    ${EL_TRAIN:+--el-train "${EL_TRAIN}"} \
    ${EL_VAL_SOURCE:+--el-val-source "${EL_VAL_SOURCE}"} \
    --val-el-n "${VAL_EL_N:-100}"
