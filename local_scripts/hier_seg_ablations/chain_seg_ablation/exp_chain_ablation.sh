#!/usr/bin/env bash
# Chain-Seg 实验: V2 (ground-seg) — 单 caption grounding + L3 分割
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_EXP_DIR="${SCRIPT_DIR}"
source "${_EXP_DIR}/../common.sh"

EXP_NAME="${EXP_NAME:-chain_seg_V2}"

# ---- 覆写 reward 为 chain_seg 专用 ----
REWARD_FUNCTION="${REPO_ROOT}/verl/reward_function/youcook2_chain_seg_reward.py:compute_score"

# ---- 原始标注 JSON 目录 & L2 clips 目录 ----
ANNOTATION_DIR="${ANNOTATION_DIR:-/m2v_intern/xuboshen/zgw/data/hier_seg_annotation/annotations}"
CLIP_DIR="${CLIP_DIR:-/m2v_intern/xuboshen/zgw/data/hier_seg_annotation/clips/L2}"

# ---- 数据目录 ----
DATA_DIR="${ABLATION_DATA_ROOT}/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

# ---- V2 单事件输出较短 ----
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-512}"

# ---- 自动数据准备 ----
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[chain_seg] Preparing V2 (ground-seg) data ..."
  python3 "${_EXP_DIR}/build_chain_seg_data.py" \
    --annotation-dir "${ANNOTATION_DIR}" \
    --clip-dir "${CLIP_DIR}" \
    --output-dir "${DATA_DIR}" \
    --total-val 100 \
    --min-events 2 \
    --complete-only
  mv "${DATA_DIR}/chain_ground_seg_train.jsonl" "${TRAIN_FILE}"
  mv "${DATA_DIR}/chain_ground_seg_val.jsonl" "${TEST_FILE}"
fi

source "${_EXP_DIR}/../launch_train.sh"
