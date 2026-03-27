#!/usr/bin/env bash
# Chain-Seg 消融: 单次实验入口
# L2L3 (原始):      多 caption grounding + L3 seg
# V1 (dual-seg):    自由双层分割, 无 caption
# V2 (ground-seg):  单 caption grounding + L3 分割
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

# ---- 原始标注 JSON 目录 & L2 clips 目录 ----
ANNOTATION_DIR="${ANNOTATION_DIR:-/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations}"
CLIP_DIR="${CLIP_DIR:-/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/clips/L2}"

# ---- 数据目录 ----
DATA_DIR="${ABLATION_DATA_ROOT}/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

# ---- V1 多事件输出更长, V2 单事件较短 ----
if [[ "${VARIANT}" == "V1" ]]; then
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
elif [[ "${VARIANT}" == "L2L3" ]]; then
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
else
  MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"
fi

# ---- 自动数据准备 (直接从原始标注 JSON 构建) ----
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[chain_seg] Preparing data for ${EXP_NAME} (variant=${VARIANT}) ..."
  python3 "${_EXP_DIR}/build_chain_seg_data.py" \
    --annotation-dir "${ANNOTATION_DIR}" \
    --clip-dir "${CLIP_DIR}" \
    --output-dir "${DATA_DIR}" \
    --variants "${VARIANT}" \
    --total-val 100 \
    --min-events 2 \
    --complete-only
  # build_chain_seg_data.py 输出 chain_{variant}_train.jsonl, 重命名为 train.jsonl
  _PREFIX_MAP_L2L3="chain_L2L3"
  _PREFIX_MAP_V1="chain_dual_seg"
  _PREFIX_MAP_V2="chain_ground_seg"
  eval "_PREFIX=\"\${_PREFIX_MAP_${VARIANT}}\""
  mv "${DATA_DIR}/${_PREFIX}_train.jsonl" "${TRAIN_FILE}"
  mv "${DATA_DIR}/${_PREFIX}_val.jsonl" "${TEST_FILE}"
fi

source "${_EXP_DIR}/../launch_train.sh"
