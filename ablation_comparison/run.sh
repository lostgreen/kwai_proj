#!/bin/bash
# Launch ablation comparison server
#
# Usage:
#   # Compare training data (different prompts, same GT):
#   bash run.sh --data
#
#   # Compare rollout outputs (different model predictions):
#   bash run.sh --rollout
#
#   # Compare V1 baseline eval (segment vs hint):
#   bash run.sh --v1-baseline
#
#   # Custom settings:
#   bash run.sh \
#     --setting PA1:/path/to/pa1/data \
#     --setting PA2:/path/to/pa2/data \
#     --port 8790

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8790}"

# Default data roots (adjust for your server)
ABLATION_DATA_ROOT="${ABLATION_DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/hier_seg_annotation/ablation_data}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/hier_seg/ablations}"
V1_EVAL_ROOT="${V1_EVAL_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation/eval_results}"

if [[ "${1:-}" == "--data" ]]; then
    echo "=== Comparing training data ==="
    python3 server.py \
        --setting "PA1:${ABLATION_DATA_ROOT}/prompt_ablation_PA1_original" \
        --setting "PA2:${ABLATION_DATA_ROOT}/prompt_ablation_PA2_v3boundary" \
        --setting "R1:${ABLATION_DATA_ROOT}/reward_ablation_R1_f1iou" \
        --setting "R2:${ABLATION_DATA_ROOT}/reward_ablation_R2_boundary" \
        --port "$PORT"
elif [[ "${1:-}" == "--rollout" ]]; then
    echo "=== Comparing rollout outputs ==="
    python3 server.py \
        --setting "PA1:${ROLLOUT_ROOT}/prompt_ablation_PA1_original/rollouts" \
        --setting "PA2:${ROLLOUT_ROOT}/prompt_ablation_PA2_v3boundary/rollouts" \
        --setting "R1:${ROLLOUT_ROOT}/reward_ablation_R1_f1iou/rollouts" \
        --setting "R2:${ROLLOUT_ROOT}/reward_ablation_R2_boundary/rollouts" \
        --port "$PORT"
elif [[ "${1:-}" == "--v1-baseline" ]]; then
    echo "=== Comparing V1 baseline: segment vs hint ==="
    SETTINGS=""
    if [[ -d "${V1_EVAL_ROOT}/segment" ]]; then
        SETTINGS="$SETTINGS --setting segment:${V1_EVAL_ROOT}/segment"
    fi
    if [[ -d "${V1_EVAL_ROOT}/segment_hint" ]]; then
        SETTINGS="$SETTINGS --setting hint:${V1_EVAL_ROOT}/segment_hint"
    fi
    if [[ -z "$SETTINGS" ]]; then
        echo "ERROR: No eval results found in ${V1_EVAL_ROOT}/"
        echo "Run eval_baseline_rollout.py first."
        exit 1
    fi
    python3 server.py $SETTINGS --port "$PORT"
else
    # Pass through all arguments
    python3 server.py "$@" --port "$PORT"
fi
