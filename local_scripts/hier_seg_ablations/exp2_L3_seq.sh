#!/usr/bin/env bash
# =============================================================
# 实验 2 — L3 Sequential Only (顺序版原子动作定位)
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-hier_seg_exp2_L3_seq}"

DATA_DIR="${HIER_DATA_ROOT}/../ablation_data/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[hier] Preparing data for ${EXP_NAME} ..."
  python3 "$(dirname "${BASH_SOURCE[0]}")/prepare_data.py" \
    --levels L3_seq \
    --total-val 200 \
    --data-root "${HIER_DATA_ROOT}" \
    --output-dir "${DATA_DIR}"
fi

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
