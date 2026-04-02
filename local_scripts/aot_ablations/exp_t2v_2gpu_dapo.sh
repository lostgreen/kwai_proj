#!/usr/bin/env bash
# =============================================================
# exp_t2v_2gpu_dapo.sh — 2卡 T2V Demo训练 + DAPO 动态过滤
#
# 用法:
#   bash local_scripts/aot_ablations/exp_t2v_2gpu_dapo.sh
#
# 核心区别（vs exp_t2v.sh）:
#   1) N_GPUS_PER_NODE=2, TP_SIZE=2 (单张卡做 TP)
#   2) ONLINE_FILTERING=true — 开启 DAPO 动态过滤
#   3) 适当缩小 batch size 以适配 2 卡
#   4) MAX_STEPS=30 用于快速 demo 验证
# =============================================================
set -euo pipefail
set -x

# ---- 2 卡配置 ----
export N_GPUS_PER_NODE=2
export TP_SIZE=2
export ROLLOUT_BS=4
export GLOBAL_BS=4
export ROLLOUT_N=8

# ---- DAPO 动态过滤 ----
export ONLINE_FILTERING=true

# ---- Demo 快速验证: 少量步数 ----
export MAX_STEPS="${MAX_STEPS:-30}"
export VAL_FREQ="${VAL_FREQ:-10}"
export SAVE_FREQ="${SAVE_FREQ:-15}"
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-phase_t2v event_t2v action_t2v}"
EXP_NAME="${EXP_NAME:-seg_aot_t2v_2gpu_dapo}"
DATA_DIR="${DATA_DIR:-${SEG_AOT_DATA_ROOT}/seg_aot_t2v}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
