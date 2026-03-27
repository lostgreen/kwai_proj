#!/usr/bin/env bash
# Chain-Seg 消融: 单次实验入口
# V1 (dual-seg): 自由双层分割, 无 caption
# V2 (ground-seg): 单 caption grounding + L3 分割
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_EXP_DIR="${SCRIPT_DIR}"
source "${_EXP_DIR}/../common.sh"

# ---- 变体选择 ----
VARIANT="${1:-${VARIANT:-V1}}"
EXP_NAME="${EXP_NAME:-chain_seg_${VARIANT}}"

# ---- 覆写 reward 为 chain_seg 专用 ----
REWARD_FUNCTION="${REPO_ROOT}/verl/reward_function/youcook2_chain_seg_reward.py:compute_score"

# ---- 数据目录 ----
DATA_DIR="${ABLATION_DATA_ROOT}/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

# ---- V1 多事件输出更长, V2 单事件较短 ----
if [[ "${VARIANT}" == "V1" ]]; then
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
else
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
fi

# ---- 自动数据准备 ----
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[chain_seg] Preparing data for ${EXP_NAME} (variant=${VARIANT}) ..."
  python3 "${_EXP_DIR}/prepare_chain_ablation_data.py" \
    --variant "${VARIANT}" \
    --total-val 100 \
    --min-events 2 \
    --data-root "${HIER_DATA_ROOT}" \
    --output-dir "${DATA_DIR}"
fi

source "${_EXP_DIR}/../launch_train.sh"
