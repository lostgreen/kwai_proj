#!/usr/bin/env bash
# =============================================================
# exp_8gpu.sh — 8卡 Grounding+Seg 训练
#
# 用法:
#   bash local_scripts/gseg_ablations/exp_8gpu.sh
# =============================================================
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP_NAME="${EXP_NAME:-gseg_8gpu}"

echo "[gseg] ── Experiment: ${EXP_NAME} (8 GPU) ──"
echo "[gseg] Train: ${TRAIN_FILE}"
echo "[gseg] Val:   ${TEST_FILE}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
