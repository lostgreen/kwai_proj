#!/usr/bin/env bash
# =============================================================
# exp_r1_f1iou.sh — Reward Ablation 实验 1: F1-IoU (Baseline)
#
# 使用现有 youcook2_hier_seg_reward.py (匈牙利匹配 + F1-IoU)
# 数据：L1 + L2 + L3 三层均衡混合
#
# 用法:
#   bash exp_r1_f1iou.sh
#   MAX_STEPS=30 bash exp_r1_f1iou.sh   # 快速调试
# =============================================================
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_EXP_DIR="${SCRIPT_DIR}"
source "${_EXP_DIR}/../common.sh"

# ---- Reward: 使用 baseline F1-IoU ----
REWARD_FUNCTION="${REPO_ROOT}/verl/reward_function/youcook2_hier_seg_reward.py:compute_score"

# ---- 实验命名 ----
EXP_NAME="${EXP_NAME:-reward_ablation_R1_f1iou}"

# ---- 数据 (L2+L3, 跳过 L1 因标注有 warped 映射问题) ----
LEVELS="L2 L3"
TRAIN_PER_LEVEL="${TRAIN_PER_LEVEL:-400}"
VAL_PER_LEVEL="${VAL_PER_LEVEL:-100}"

DATA_DIR="${ABLATION_DATA_ROOT}/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

# ---- 数据构建（首次自动触发）----
_LEVELS_TAG="$(echo "${LEVELS}" | tr ' ' '_')"
BASE_DATA_DIR="${ABLATION_DATA_ROOT}/hier_seg_base_${_LEVELS_TAG}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  # Step 1: 基础数据
  if [[ ! -f "${BASE_DATA_DIR}/train.jsonl" ]]; then
    echo "[reward_ablation] Step 1: Building base data for levels: ${LEVELS} ..."
    BUILD_LEVELS="${LEVELS//L3/L3_seg}"
    # shellcheck disable=SC2086
    python3 "${_EXP_DIR}/../build_hier_data.py" \
      --annotation-dir "${ANNOTATION_DIR}" \
      --clip-dir-l2 "${CLIP_DIR_L2}" \
      --clip-dir-l3 "${CLIP_DIR_L3}" \
      --output-dir "${BASE_DATA_DIR}" \
      --levels ${BUILD_LEVELS} \
      --total-val "$(( VAL_PER_LEVEL * 2 ))" \
      --train-per-level "${TRAIN_PER_LEVEL}" \
      --complete-only
  fi

  # Step 2: 使用 V3 prompt V2 变体 (边界判据+稀疏采样感知，控制 prompt 变量)
  echo "[reward_ablation] Step 2: Applying V3-prompt V2 variant (boundary-criterion, controlled variable) ..."
  # shellcheck disable=SC2086
  python3 "${_EXP_DIR}/../prompt_ablation/prepare_v2_ablation_data.py" \
    --levels ${LEVELS} \
    --variant V2 \
    --prompt-version v3 \
    --val-per-level "${VAL_PER_LEVEL}" \
    --train-per-level "${TRAIN_PER_LEVEL}" \
    --data-root "${BASE_DATA_DIR}" \
    --output-dir "${DATA_DIR}"
fi

# ---- 启动训练 ----
source "${_EXP_DIR}/../launch_train.sh"
