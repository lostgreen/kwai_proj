#!/usr/bin/env bash
# =============================================================
# 实验 1 — L2 Only (滑窗事件检测)
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-hier_seg_exp1_L2_only}"

DATA_DIR="${HIER_DATA_ROOT}/../ablation_data/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[hier] Preparing data for ${EXP_NAME} ..."
  python3 "$(dirname "${BASH_SOURCE[0]}")/prepare_data.py" \
    --levels L2 \
    --val-per-level 20 \
    --data-root "${HIER_DATA_ROOT}" \
    --output-dir "${DATA_DIR}"
fi

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
