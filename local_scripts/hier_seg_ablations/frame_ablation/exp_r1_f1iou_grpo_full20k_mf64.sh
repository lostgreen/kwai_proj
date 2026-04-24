#!/usr/bin/env bash
# =============================================================
# R1 F1-IoU / GRPO frame ablation: max_frames=64.
#
# Hier input semantics:
#   L1: full video
#   L2: phase-crop clip, events zero-based to phase start
#   L3: event-crop clip
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_LOCAL="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

PHASECROP_ROOT="${PHASECROP_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/train_phasecrop}"
DEFAULT_HIER_TRAIN_PHASECROP="${PHASECROP_ROOT}/train_all_shared_frames.jsonl"
DEFAULT_HIER_VAL_PHASECROP="${PHASECROP_ROOT}/val_all_shared_frames.jsonl"

export EXP_NAME="${EXP_NAME:-frame_ablation_R1_f1iou_grpo_full20k_mf64}"
export TASKS="${TASKS:-tg mcq hier_seg}"
export HIER_TARGET="${HIER_TARGET:-0}"

if [[ "${ALLOW_HIER_TRAIN_OVERRIDE:-false}" =~ ^(true|1|yes)$ ]]; then
    export HIER_TRAIN="${HIER_TRAIN:-${DEFAULT_HIER_TRAIN_PHASECROP}}"
    export HIER_VAL_SOURCE="${HIER_VAL_SOURCE:-${DEFAULT_HIER_VAL_PHASECROP}}"
else
    if [[ -n "${HIER_TRAIN:-}" && "${HIER_TRAIN}" != "${DEFAULT_HIER_TRAIN_PHASECROP}" ]]; then
        echo "[frame_ablation mf64] Ignore inherited HIER_TRAIN=${HIER_TRAIN}; using phase-crop manifest." >&2
    fi
    if [[ -n "${HIER_VAL_SOURCE:-}" && "${HIER_VAL_SOURCE}" != "${DEFAULT_HIER_VAL_PHASECROP}" ]]; then
        echo "[frame_ablation mf64] Ignore inherited HIER_VAL_SOURCE=${HIER_VAL_SOURCE}; using phase-crop manifest." >&2
    fi
    export HIER_TRAIN="${DEFAULT_HIER_TRAIN_PHASECROP}"
    export HIER_VAL_SOURCE="${DEFAULT_HIER_VAL_PHASECROP}"
fi

if [[ ! -f "${HIER_TRAIN}" || ! -f "${HIER_VAL_SOURCE}" ]]; then
    echo "[frame_ablation mf64] Missing phase-crop shared manifest." >&2
    echo "  HIER_TRAIN=${HIER_TRAIN}" >&2
    echo "  HIER_VAL_SOURCE=${HIER_VAL_SOURCE}" >&2
    echo "Run: bash local_scripts/hier_seg_ablations/frame_ablation/build_phasecrop_shared_hier.sh" >&2
    exit 1
fi

if [[ "${CHECK_PHASECROP_MANIFEST:-true}" =~ ^(true|1|yes)$ ]]; then
    python3 "${REPO_ROOT_LOCAL}/local_scripts/data/check_hier_phasecrop_manifest.py" \
        --jsonl "${HIER_TRAIN}" \
        --jsonl "${HIER_VAL_SOURCE}"
fi

export ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
export ONLINE_FILTERING="${ONLINE_FILTERING:-true}"
export LR="${LR:-1e-6}"
export KL_COEF="${KL_COEF:-0.001}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export MAX_FRAMES="${MAX_FRAMES:-64}"
export FRAME_SAMPLE_MAX_FRAMES="${FRAME_SAMPLE_MAX_FRAMES:-64}"
export FRAME_SAMPLE_POLICY="${FRAME_SAMPLE_POLICY:-0:32:2.0,32:inf:1.0}"
export MAX_PIXELS="${MAX_PIXELS:-65536}"
export CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
export CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"

export HIER_REWARD_MODE="${HIER_REWARD_MODE:-f1_iou}"
export REWARD_FUNCTION="${REWARD_FUNCTION:-${REPO_ROOT_LOCAL}/verl/reward_function/mixed_proxy_reward_ablation.py:compute_score}"

source "${REPO_ROOT_LOCAL}/local_scripts/run_multi_task.sh"
