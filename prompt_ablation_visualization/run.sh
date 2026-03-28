#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Configuration ──────────────────────────────────────────────────
# Override these via environment variables or edit defaults below.

VIZHOST="${VIZHOST:-0.0.0.0}"
PORT="${PORT:-8891}"

# Experiment rollout directories (comma-separated V=path pairs)
# Example: V1=/path/to/V1/rollouts,V2=/path/to/V2/rollouts,...
MODEL_ROOT="${MODEL_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/hier_seg/ablations}"

EXP_DIRS="${EXP_DIRS:-V1=${MODEL_ROOT}/hier_seg_v2_V1_L1_L2_L3/rollouts,V2=${MODEL_ROOT}/hier_seg_v2_V2_L1_L2_L3/rollouts,V3=${MODEL_ROOT}/hier_seg_v2_V3_L1_L2_L3/rollouts,V4=${MODEL_ROOT}/hier_seg_v2_V4_L1_L2_L3/rollouts}"

# Log files (optional, comma-separated V=path pairs)
LOG_FILES="${LOG_FILES:-V1=${MODEL_ROOT}/hier_seg_v2_V1_L1_L2_L3/experiment_log.jsonl,V2=${MODEL_ROOT}/hier_seg_v2_V2_L1_L2_L3/experiment_log.jsonl,V3=${MODEL_ROOT}/hier_seg_v2_V3_L1_L2_L3/experiment_log.jsonl,V4=${MODEL_ROOT}/hier_seg_v2_V4_L1_L2_L3/experiment_log.jsonl}"

# ── Launch ─────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════╗"
echo "║  Prompt Ablation Comparison Studio           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "[config] host=${VIZHOST} port=${PORT}"
echo "[config] exp_dirs=${EXP_DIRS}"
echo "[config] log_files=${LOG_FILES}"
echo ""
echo "[open] http://localhost:${PORT}/"
echo ""

cd "${REPO_ROOT}"
python prompt_ablation_visualization/server.py \
  --host "${VIZHOST}" \
  --port "${PORT}" \
  --static-dir prompt_ablation_visualization \
  --exp-dirs "${EXP_DIRS}" \
  --log-files "${LOG_FILES}"
