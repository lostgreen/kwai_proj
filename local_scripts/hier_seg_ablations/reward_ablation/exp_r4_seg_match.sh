#!/usr/bin/env bash
# =============================================================
# exp_r4_seg_match.sh — Reward Ablation: Segment Matching
#
# 多任务混合: LLaVA MCQ + TG + Hier Seg
# Reward: seg_match_reward for HierSeg (range [0, 1])
#         mixed_proxy_reward handles MCQ → choice, TG → tIoU
#
# TODO: 需要在 mixed_proxy_reward.py 注册 seg_match dispatch,
#       或创建 wrapper 组合 seg_match + choice + tIoU
# =============================================================
set -euo pipefail

# ---- 实验特有配置 ----
export EXP_NAME="${EXP_NAME:-reward_ablation_R4_seg_match}"

# ---- 启用的任务 + 数据量 (与 R1 相同，控制变量) ----
export TASKS="${TASKS:-tg mcq hier_seg}"
export HIER_TARGET="${HIER_TARGET:-2000}"

# ---- Reward: Segment Matching ----
# TODO: 当前 mixed_proxy_reward 对 hier_seg 用 F1-IoU,
#       需要添加 seg_match dispatch 后取消注释
# export REWARD_FUNCTION="${REPO_ROOT}/verl/reward_function/mixed_proxy_reward.py:compute_score"

# ---- 启动 ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../run_multi_task.sh"
