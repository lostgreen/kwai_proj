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
ABLATION_DATA_ROOT="${ABLATION_DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/ablation_data}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-/m2v_intern/xuboshen/zgw/hier_seg/ablations}"

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
    # Rollout dirs are inside each experiment's checkpoint directory
    python3 server.py \
        --setting "PA1:${ROLLOUT_ROOT}/prompt_ablation_PA1_original/rollout" \
        --setting "PA2:${ROLLOUT_ROOT}/prompt_ablation_PA2_v3boundary/rollout" \
        --setting "R1:${ROLLOUT_ROOT}/reward_ablation_R1_f1iou/rollout" \
        --setting "R2:${ROLLOUT_ROOT}/reward_ablation_R2_boundary/rollout" \
        --port "$PORT"
else
    # Pass through all arguments
    python3 server.py "$@" --port "$PORT"
fi
