#!/usr/bin/env bash
# =============================================================
# exp_r2_boundary.sh — Reward Ablation 实验 2: Boundary-Aware Reward
#
# 使用 youcook2_hier_seg_reward_boundary.py:
#   0.5 × Boundary Hit F1 + 0.2 × Count Accuracy + 0.3 × Coverage IoU
#
# 数据和 prompt 与 exp_r1 完全对齐，仅改变 reward function。
#
# 用法:
#   bash exp_r2_boundary.sh
#   MAX_STEPS=30 bash exp_r2_boundary.sh   # 快速调试
# =============================================================
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_EXP_DIR="${SCRIPT_DIR}"
source "${_EXP_DIR}/../common.sh"

# ---- Reward: 使用 Boundary-Aware 变体 ----
REWARD_FUNCTION="${REPO_ROOT}/verl/reward_function/youcook2_hier_seg_reward_boundary.py:compute_score"

# ---- 实验命名 ----
EXP_NAME="${EXP_NAME:-reward_ablation_R2_boundary}"

# ---- 数据 (与 R1 完全相同，L2+L3，控制变量) ----
LEVELS="L2 L3"
TRAIN_PER_LEVEL="${TRAIN_PER_LEVEL:-400}"
VAL_PER_LEVEL="${VAL_PER_LEVEL:-100}"

# 复用 R1 的数据（确保完全相同的训练/验证集）
R1_DATA_DIR="${ABLATION_DATA_ROOT}/reward_ablation_R1_f1iou"
DATA_DIR="${ABLATION_DATA_ROOT}/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

# ---- 数据构建 ----
_LEVELS_TAG="$(echo "${LEVELS}" | tr ' ' '_')"
BASE_DATA_DIR="${ABLATION_DATA_ROOT}/hier_seg_base_${_LEVELS_TAG}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  # 优先复用 R1 数据（保证完全相同）
  if [[ -f "${R1_DATA_DIR}/train.jsonl" ]]; then
    echo "[reward_ablation] Reusing R1 data for controlled comparison ..."
    mkdir -p "${DATA_DIR}"
    cp "${R1_DATA_DIR}/train.jsonl" "${TRAIN_FILE}"
    cp "${R1_DATA_DIR}/val.jsonl" "${TEST_FILE}"
  else
    # R1 未运行，自行构建
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
fi

# ---- 启动训练 ----
source "${_EXP_DIR}/../launch_train.sh"
