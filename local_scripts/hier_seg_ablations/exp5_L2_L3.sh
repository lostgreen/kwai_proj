#!/usr/bin/env bash
# =============================================================
# 实验 5 — L2 + L3 Sequential (事件检测 + 动作定位联合)
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-hier_seg_exp5_L2_L3}"

DATA_DIR="${ABLATION_DATA_ROOT}/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[hier] Preparing data for ${EXP_NAME} ..."
  python3 "$(dirname "${BASH_SOURCE[0]}")/prepare_data.py" \
    --levels L2 L3_seq \
    --total-val 200 \
    --data-root "${HIER_DATA_ROOT}" \
    --output-dir "${DATA_DIR}"
fi

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
