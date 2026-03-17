#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
# Guard against conda cross-compile HOST pollution (e.g. x86_64-conda-linux-gnu)
_raw_host="${HOST:-0.0.0.0}"
if [[ "$_raw_host" != "0.0.0.0" && "$_raw_host" != "127.0.0.1" && ! "$_raw_host" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  HOST="0.0.0.0"
else
  HOST="$_raw_host"
fi
PORT="${PORT:-8890}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PREFER_COMPLETE="${PREFER_COMPLETE:-1}"

# DATA_PATH takes priority: supports annotation dir, single JSON, or built dataset JSONL.
# Falls back to ANNOTATION_DIR for backward compatibility.
if [[ -n "${DATA_PATH:-}" ]]; then
  LOAD_PATH="${DATA_PATH}"
else
  LOAD_PATH="${ANNOTATION_DIR:-proxy_data/youcook2_seg_annotation/annotations}"
fi

echo "[seg-viz] host=${HOST} port=${PORT}"
echo "[seg-viz] data_path=${LOAD_PATH}"
echo "[seg-viz] max_samples=${MAX_SAMPLES} prefer_complete=${PREFER_COMPLETE}"
echo "[seg-viz] open: http://localhost:${PORT}/"

cd "${REPO_ROOT}"
ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --static-dir "${ROOT_DIR}"
  --data-path "${LOAD_PATH}"
  --max-samples "${MAX_SAMPLES}"
)

if [[ "${PREFER_COMPLETE}" == "1" ]]; then
  ARGS+=(--prefer-complete)
fi

python "${ROOT_DIR}/server.py" \
  "${ARGS[@]}"
