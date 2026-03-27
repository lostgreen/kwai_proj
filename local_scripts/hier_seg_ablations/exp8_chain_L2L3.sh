#!/usr/bin/env bash
# =============================================================
# 实验 8 — Chain-of-Segment (L2 Grounding + L3 Atomic Segmentation)
#
# 链式层次分割：模型先定位 L2 事件，再在每个事件内分割 L3 原子动作。
# 使用专用 reward: youcook2_chain_seg_reward.py
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-hier_seg_exp8_chain_L2L3}"

# ---- 使用 Chain-of-Segment 专用 reward ----
REWARD_FUNCTION="${REPO_ROOT}/verl/reward_function/youcook2_chain_seg_reward.py:compute_score"

# ---- 数据目录 ----
DATA_DIR="${ABLATION_DATA_ROOT}/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

# ---- 加大 response 长度（L2+L3 嵌套输出更长）----
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[hier] Preparing Chain-of-Segment data for ${EXP_NAME} ..."
  python3 "$(dirname "${BASH_SOURCE[0]}")/prepare_chain_seg_data.py" \
    --total-val 100 \
    --min-events 2 \
    --data-root "${HIER_DATA_ROOT}" \
    --output-dir "${DATA_DIR}"
fi

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
