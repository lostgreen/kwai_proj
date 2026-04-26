#!/usr/bin/env bash
# =============================================================
# exp_r1_f1iou_ema.sh — Reward Ablation: R1 / Hungarian F1-IoU / EMA-GRPO
#
# 保险对照实验:
#   - 与 R1-GRPO 保持同一套 reward / 数据 / 超参
#   - 只把 ADV_ESTIMATOR 切回 ema_grpo
#   - ROLLOUT_TEMPERATURE=1.0
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_LOCAL="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${REPO_ROOT_LOCAL}/local_scripts/ablation_common.sh"

# ---- 实验特有配置 ----
export EXP_NAME="${EXP_NAME:-reward_ablation_R1_f1iou_ema_grpo_full20k}"

# ---- 启用的任务 + 数据量 ----
export TASKS="${TASKS:-tg mcq hier_seg}"
export HIER_TARGET="${HIER_TARGET:-0}"

# ---- 优先走共享 frame-list manifest；不存在时回退到全量 jsonl ----
DEFAULT_HIER_TRAIN_SHARED="/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/train/train_all_shared_frames.jsonl"
DEFAULT_HIER_VAL_SHARED="/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/train/val_all_shared_frames.jsonl"
if [[ -z "${HIER_TRAIN:-}" && -f "${DEFAULT_HIER_TRAIN_SHARED}" ]]; then
    export HIER_TRAIN="${DEFAULT_HIER_TRAIN_SHARED}"
fi
if [[ -z "${HIER_VAL_SOURCE:-}" && -f "${DEFAULT_HIER_VAL_SHARED}" ]]; then
    export HIER_VAL_SOURCE="${DEFAULT_HIER_VAL_SHARED}"
fi

# ---- 关键超参（EMA-GRPO + online filtering） ----
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-ema_grpo}"
export ONLINE_FILTERING="${ONLINE_FILTERING:-true}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0.005}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export MAX_FRAMES="${MAX_FRAMES:-256}"
export MAX_PIXELS="${MAX_PIXELS:-65536}"
export CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
export CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"

# ---- Reward: 显式切到 hier F1-IoU ----
export HIER_REWARD_MODE="${HIER_REWARD_MODE:-f1_iou}"
export REWARD_FUNCTION="${REWARD_FUNCTION:-${REPO_ROOT_LOCAL}/verl/reward_function/mixed_proxy_reward_ablation.py:compute_score}"

# ---- 启动 ----
source "${SCRIPT_DIR}/../../run_multi_task.sh"
