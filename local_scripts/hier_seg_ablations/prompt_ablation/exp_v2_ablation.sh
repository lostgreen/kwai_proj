#!/usr/bin/env bash
# =============================================================
# exp_v2_ablation.sh — V2 通用 Prompt 消融实验
#
# 支持 L1 / L2 / L3 任意层级组合，Variant V1-V4
# 默认采样: 每层 400 train + 100 val（均衡三层 = 1200 train / 300 val）
#
# 用法:
#   # 三层均衡（默认 400+100 per level）
#   VARIANT=V1 LEVELS="L1 L2 L3" bash exp_v2_ablation.sh
#
#   # 仅 L2，V2 variant
#   VARIANT=V2 LEVELS="L2" bash exp_v2_ablation.sh
#
#   # 自定义采样量（-1 = 不限制）
#   TRAIN_PER_LEVEL=200 VAL_PER_LEVEL=50 VARIANT=V1 LEVELS="L1 L2 L3" bash exp_v2_ablation.sh
#
#   # 也可直接传参: bash exp_v2_ablation.sh V3 "L1 L2 L3"
# =============================================================
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# 保存自身目录（common.sh 会覆盖 SCRIPT_DIR，这里提前存好）
_EXP_DIR="${SCRIPT_DIR}"
source "${_EXP_DIR}/../common.sh"

VARIANT="${1:-${VARIANT:-V1}}"
LEVELS="${2:-${LEVELS:-L1 L2 L3}}"  # 空格分隔的层级列表

# 每层数据量（默认三层均衡: 400 train + 100 val = 1200/300 总计）
VAL_PER_LEVEL="${VAL_PER_LEVEL:-100}"
TRAIN_PER_LEVEL="${TRAIN_PER_LEVEL:-400}"

# ---- 实验命名 ----
_LEVELS_TAG="$(echo "${LEVELS}" | tr ' ' '_')"
EXP_NAME="${EXP_NAME:-hier_seg_v2_${VARIANT}_${_LEVELS_TAG}}"

DATA_DIR="${ABLATION_DATA_ROOT}/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

# V3/V4 需要更长的 response（含 <think>）
if [[ "${VARIANT}" == "V3" || "${VARIANT}" == "V4" ]]; then
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
fi

# ---- 数据准备（首次自动触发）----
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[hier] Preparing V2/${VARIANT} data for levels: ${LEVELS} ..."
  # shellcheck disable=SC2086
  python3 "${_EXP_DIR}/prepare_v2_ablation_data.py" \
    --levels ${LEVELS} \
    --variant "${VARIANT}" \
    --val-per-level "${VAL_PER_LEVEL}" \
    --train-per-level "${TRAIN_PER_LEVEL}" \
    --data-root "${HIER_DATA_ROOT}" \
    --output-dir "${DATA_DIR}"
fi

# ---- 启动训练 ----
source "${_EXP_DIR}/../launch_train.sh"
