#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8765}"
ROLLOUT_DIR="${ROLLOUT_DIR:-checkpoints/qwen3_vl_mixed_proxy_training/rollouts}"
LOG_FILE="${LOG_FILE:-checkpoints/qwen3_vl_mixed_proxy_training/experiment_log.jsonl}"

echo "[rollout-viz] host=${HOST} port=${PORT}"
echo "[rollout-viz] default rollout_dir=${ROLLOUT_DIR}"
echo "[rollout-viz] default log_file=${LOG_FILE}"
echo "[rollout-viz] open:"
echo "http://localhost:${PORT}/?rollout_dir=${ROLLOUT_DIR}&log_file=${LOG_FILE}&autoload=1"

cd "${REPO_ROOT}"
python rollout_visualization/server.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --static-dir rollout_visualization
