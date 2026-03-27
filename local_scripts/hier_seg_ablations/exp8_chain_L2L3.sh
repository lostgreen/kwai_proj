#!/usr/bin/env bash
# =============================================================
# 实验 8 — Chain-of-Segment (L2 Grounding + L3 Atomic Segmentation)
#
# 链式层次分割：模型先定位 L2 事件，再在每个事件内分割 L3 原子动作。
# 使用专用 reward: youcook2_chain_seg_reward.py
#
# 已迁移至 build_chain_seg_data.py，直接从原始标注 JSON 构建。
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-hier_seg_exp8_chain_L2L3}"

# ---- 使用 Chain-of-Segment 专用 reward ----
REWARD_FUNCTION="${REPO_ROOT}/verl/reward_function/youcook2_chain_seg_reward.py:compute_score"

# ---- 原始标注 JSON 目录 & L2 clips 目录 ----
ANNOTATION_DIR="${ANNOTATION_DIR:-/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations}"
CLIP_DIR="${CLIP_DIR:-/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/clips/L2}"

# ---- 数据目录 ----
DATA_DIR="${ABLATION_DATA_ROOT}/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

# ---- 加大 response 长度（L2+L3 嵌套输出更长）----
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[hier] Preparing Chain-of-Segment data for ${EXP_NAME} ..."
  python3 "$(dirname "${BASH_SOURCE[0]}")/chain_seg_ablation/build_chain_seg_data.py" \
    --annotation-dir "${ANNOTATION_DIR}" \
    --clip-dir "${CLIP_DIR}" \
    --output-dir "${DATA_DIR}" \
    --variants L2L3 \
    --total-val 100 \
    --min-events 2 \
    --complete-only
  # 重命名以适配通用 launch_train.sh
  mv "${DATA_DIR}/chain_L2L3_train.jsonl" "${TRAIN_FILE}"
  mv "${DATA_DIR}/chain_L2L3_val.jsonl" "${TEST_FILE}"
fi

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
