#!/usr/bin/env bash
# Shared defaults for task-composition ablations:
#   base-only and base plus seg/aot/event-logic combinations.

COMPOSITION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_LOCAL="$(cd -- "${COMPOSITION_DIR}/../.." && pwd)"

source "${REPO_ROOT_LOCAL}/local_scripts/ablation_common.sh"

export PROJECT_NAME="${PROJECT_NAME:-EasyR1-task-composition-ablation}"

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

export HIER_TARGET="${HIER_TARGET:-10000}"
DEFAULT_HIER_TRAIN_SHARED="/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/train/train_all_shared_frames.jsonl"
DEFAULT_HIER_VAL_SHARED="/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/train/val_all_shared_frames.jsonl"
if [[ -z "${HIER_TRAIN:-}" && -f "${DEFAULT_HIER_TRAIN_SHARED}" ]]; then
    export HIER_TRAIN="${DEFAULT_HIER_TRAIN_SHARED}"
fi
if [[ -z "${HIER_VAL_SOURCE:-}" && -f "${DEFAULT_HIER_VAL_SHARED}" ]]; then
    export HIER_VAL_SOURCE="${DEFAULT_HIER_VAL_SHARED}"
fi

export AOT_TRAIN="${AOT_TRAIN:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/aot/train_nocot_reward_balanced.jsonl}"
export AOT_TARGET="${AOT_TARGET:-10000}"
export VAL_AOT_N="${VAL_AOT_N:-300}"
