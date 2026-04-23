#!/usr/bin/env bash
# =============================================================
# exp_r1_f1iou.sh — Reward Ablation: F1-IoU (Baseline)
#
# 多任务混合: LLaVA MCQ + TG + Hier Seg
# Reward: mixed_proxy_reward (HierSeg → F1-IoU, MCQ → choice, TG → tIoU)
#
# 用法:
#   bash exp_r1_f1iou.sh                    # 2卡默认
#   N_GPUS_PER_NODE=8 ROLLOUT_BS=16 GLOBAL_BS=16 \
#     bash exp_r1_f1iou.sh                  # 8卡
#   MAX_STEPS=30 bash exp_r1_f1iou.sh       # 快速调试
# =============================================================
set -euo pipefail

# ---- 实验特有配置 ----
export EXP_NAME="${EXP_NAME:-reward_ablation_R1_f1iou}"

# ---- 启用的任务 + 数据量 ----
export TASKS="${TASKS:-tg mcq hier_seg}"
export HIER_TARGET="${HIER_TARGET:-2000}"
# export EL_TARGET="${EL_TARGET:-2000}"   # 如需 event_logic, 取消注释并加入 TASKS

# ---- Reward: F1-IoU (mixed_proxy_reward 默认已注册) ----
# 无需覆盖 REWARD_FUNCTION

# ---- 启动 (source common + 数据构建 + 训练) ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../run_multi_task.sh"
