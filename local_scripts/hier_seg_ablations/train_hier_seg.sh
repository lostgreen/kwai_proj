#!/usr/bin/env bash
# =============================================================
# train_hier_seg.sh — 三层层级分割训练（L1 + L2 + L3）
#
# 使用 NGIoU V2 reward，数据来自 build_v1_data.sh 的输出。
#
# 用法:
#   # 默认：读取已构建的 hier_seg_annotation/train/ 数据
#   bash train_hier_seg.sh
#
#   # 指定数据目录
#   DATA_DIR=/path/to/train bash train_hier_seg.sh
#
#   # 快速调试（少量步数）
#   MAX_STEPS=10 bash train_hier_seg.sh
#
#   # 使用 hint 版本数据
#   USE_HINT=true bash train_hier_seg.sh
#
# 前置条件:
#   build_v1_data.sh 已运行完成，生成了 train_all.jsonl / val_all.jsonl
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# ---- 实验命名 ----
EXP_NAME="${EXP_NAME:-hier_seg_L1_L2_L3_ngiou}"

# ---- 数据路径 ----
DATA_DIR="${DATA_DIR:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation/train}"

# hint 模式
USE_HINT="${USE_HINT:-false}"
if [[ "${USE_HINT}" == "true" ]]; then
    SUFFIX="_hint"
else
    SUFFIX=""
fi

TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train${SUFFIX}_all.jsonl}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/val${SUFFIX}_all.jsonl}"

# ---- Reward: NGIoU V2 (已默认指向 youcook2_hier_seg_reward.py) ----
# common.sh 中 REWARD_FUNCTION 已指向 youcook2_hier_seg_reward.py:compute_score
# _DISPATCH 已更新为 NGIoU 版本 (_l1_reward / _l2_reward / _l3_reward_v2)

# ---- 检查数据是否存在 ----
if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "[train] ERROR: Train file not found: ${TRAIN_FILE}" >&2
    echo "[train] Please run build_v1_data.sh first:" >&2
    echo "[train]   bash proxy_data/youcook2_seg/hier_seg_annotation/build_v1_data.sh" >&2
    exit 1
fi

echo "[train] ============================================"
echo "[train]  Hier Seg Training — L1 + L2 + L3 (NGIoU)"
echo "[train] ============================================"
echo "[train]  EXP_NAME:  ${EXP_NAME}"
echo "[train]  DATA_DIR:  ${DATA_DIR}"
echo "[train]  HINT:      ${USE_HINT}"
echo "[train]  REWARD:    ${REWARD_FUNCTION}"

# ---- 启动训练 ----
source "${SCRIPT_DIR}/launch_train.sh"
