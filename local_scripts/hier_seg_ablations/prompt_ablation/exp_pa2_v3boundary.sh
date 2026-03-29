#!/usr/bin/env bash
# =============================================================
# exp_pa2_v3boundary.sh — Prompt Ablation 实验 2: V3 边界判据 Prompt
#
# 使用 prompt_variants_v3.py V2 变体 (Enhanced Hard Rules):
#   L2: "LOCAL TASK UNIT" + 边界标准 + 稀疏采样约束 + 硬规则
#   L3: "VISIBLE STATE-CHANGE" + min/max duration + merge/split 规则
#
# 特点: 领域无关，边界判据导向，显式稀疏采样感知
#
# 用法:
#   bash exp_pa2_v3boundary.sh
#   MAX_STEPS=30 bash exp_pa2_v3boundary.sh
# =============================================================
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_EXP_DIR="${SCRIPT_DIR}"
source "${_EXP_DIR}/../../common.sh"

# ---- 实验命名 ----
EXP_NAME="${EXP_NAME:-prompt_ablation_PA2_v3boundary}"

# ---- 数据: L2+L3 (跳过 L1 warped 问题) ----
LEVELS="L2 L3"
TRAIN_PER_LEVEL="${TRAIN_PER_LEVEL:-400}"
VAL_PER_LEVEL="${VAL_PER_LEVEL:-100}"

DATA_DIR="${ABLATION_DATA_ROOT}/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

_LEVELS_TAG="$(echo "${LEVELS}" | tr ' ' '_')"
BASE_DATA_DIR="${ABLATION_DATA_ROOT}/hier_seg_base_${_LEVELS_TAG}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  # Step 1: 基础数据（复用 PA1 的 build_hier_data.py 输出）
  if [[ ! -f "${BASE_DATA_DIR}/train.jsonl" ]]; then
    echo "[prompt_ablation] Step 1: Building base data from annotations ..."
    BUILD_LEVELS="${LEVELS//L3/L3_seg}"
    # shellcheck disable=SC2086
    python3 "${_EXP_DIR}/../../build_hier_data.py" \
      --annotation-dir "${ANNOTATION_DIR}" \
      --clip-dir-l2 "${CLIP_DIR_L2}" \
      --clip-dir-l3 "${CLIP_DIR_L3}" \
      --output-dir "${BASE_DATA_DIR}" \
      --levels ${BUILD_LEVELS} \
      --total-val "$(( VAL_PER_LEVEL * 2 ))" \
      --train-per-level "${TRAIN_PER_LEVEL}" \
      --complete-only
  fi

  # Step 2: 替换为 V3 prompt V2 变体 (边界判据 + 稀疏采样 + 硬规则)
  echo "[prompt_ablation] Step 2: Applying V3-prompt V2 variant (boundary-criterion + sparse-aware) ..."
  # shellcheck disable=SC2086
  python3 "${_EXP_DIR}/prepare_prompt_data.py" \
    --levels ${LEVELS} \
    --variant V2 \
    --val-per-level "${VAL_PER_LEVEL}" \
    --train-per-level "${TRAIN_PER_LEVEL}" \
    --data-root "${BASE_DATA_DIR}" \
    --output-dir "${DATA_DIR}"
fi

# ---- 启动训练 ----
source "${_EXP_DIR}/../../launch_train.sh"
