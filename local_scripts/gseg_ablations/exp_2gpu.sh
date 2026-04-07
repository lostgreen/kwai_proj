#!/usr/bin/env bash
# =============================================================
# exp_2gpu.sh — 2卡 Grounding+Seg 训练 (快速验证)
#
# 核心区别（vs exp_8gpu.sh）:
#   1) N_GPUS_PER_NODE=2, TP_SIZE=2
#   2) 缩小 batch size 适配 2 卡
#   3) MAX_STEPS=30 用于快速 demo 验证
#
# 用法:
#   bash local_scripts/gseg_ablations/exp_2gpu.sh
# =============================================================
set -euo pipefail

# ---- 2 卡配置 ----
export N_GPUS_PER_NODE=2
export TP_SIZE=2
export ROLLOUT_BS=4
export GLOBAL_BS=4
export ROLLOUT_N=8

# ---- 快速验证 ----
export MAX_STEPS="${MAX_STEPS:-30}"
export VAL_FREQ="${VAL_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-15}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXP_NAME="${EXP_NAME:-gseg_2gpu}"

echo "[gseg] ── Experiment: ${EXP_NAME} (2 GPU demo) ──"
echo "[gseg] Train: ${TRAIN_FILE}"
echo "[gseg] Val:   ${TEST_FILE}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
