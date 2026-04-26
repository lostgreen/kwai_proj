#!/usr/bin/env bash
# =============================================================
# exp_base_aot.sh — TG + MCQ base mixed with Temporal AoT.
#
# Default data:
#   AOT_TRAIN=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/aot/train_nocot_reward_balanced.jsonl
#   AOT_TARGET=10000
#
# Usage:
#   bash local_scripts/aot_ablations/exp_base_aot.sh
#   MIX_ONLY=true MIX_FORCE=true bash local_scripts/aot_ablations/exp_base_aot.sh
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_LOCAL="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT_LOCAL}/local_scripts/ablation_common.sh"

export EXP_NAME="${EXP_NAME:-aot_base_tg3k_llava0125_aot10k_mf256}"
export TASKS="${TASKS:-tg mcq aot}"
export AOT_TRAIN="${AOT_TRAIN:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/aot/train_nocot_reward_balanced.jsonl}"
export AOT_TARGET="${AOT_TARGET:-10000}"
export VAL_AOT_N="${VAL_AOT_N:-300}"

export MAX_FRAMES="${MAX_FRAMES:-256}"
export FRAME_SAMPLE_MAX_FRAMES="${FRAME_SAMPLE_MAX_FRAMES:-256}"
export FRAME_SAMPLE_POLICY="${FRAME_SAMPLE_POLICY:-0:60:2.0,60:inf:1.0}"

export ADV_ESTIMATOR="${ADV_ESTIMATOR:-ema_grpo}"
export ONLINE_FILTERING="${ONLINE_FILTERING:-true}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export MAX_PIXELS="${MAX_PIXELS:-65536}"
export CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
export CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"
export REWARD_FUNCTION="${REWARD_FUNCTION:-${REPO_ROOT_LOCAL}/verl/reward_function/mixed_proxy_reward.py:compute_score}"

source "${SCRIPT_DIR}/../run_multi_task.sh"
