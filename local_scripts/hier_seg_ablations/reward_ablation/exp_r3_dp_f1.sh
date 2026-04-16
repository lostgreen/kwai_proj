#!/usr/bin/env bash
# =============================================================
# exp_r3_dp_f1.sh — Reward Ablation: DP-F1 + Instance Count
#
# 多任务混合: LLaVA MCQ + TG + Hier Seg
# Reward: dp_f1_reward for HierSeg (range [0, 2])
#         mixed_proxy_reward handles MCQ → choice, TG → tIoU
#
# TODO: 需要在 mixed_proxy_reward.py 注册 dp_f1 dispatch,
#       或创建 wrapper 组合 dp_f1 + choice + tIoU
# =============================================================
set -euo pipefail

# ---- 实验特有配置 ----
export EXP_NAME="${EXP_NAME:-reward_ablation_R3_dp_f1}"

# ---- 启用的任务 + 数据量 (与 R1 相同，控制变量) ----
export TASKS="${TASKS:-tg mcq hier_seg}"
export HIER_TARGET="${HIER_TARGET:-5000}"

# ---- Reward: DP-F1 ----
# TODO: 当前 mixed_proxy_reward 对 hier_seg 用 F1-IoU,
#       需要添加 dp_f1 dispatch 后取消注释
# export REWARD_FUNCTION="${REPO_ROOT}/verl/reward_function/mixed_proxy_reward.py:compute_score"

# ---- Online filtering 阈值适配 [0, 2] 范围 ----
export FILTER_LOW="${FILTER_LOW:-0.3}"
export FILTER_HIGH="${FILTER_HIGH:-1.6}"

# ---- 启动 ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../run_multi_task.sh"
