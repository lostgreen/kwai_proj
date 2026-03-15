#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8890}"
ANNOTATION_DIR="${ANNOTATION_DIR:-proxy_data/youcook2_seg_annotation/annotations}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PREFER_COMPLETE="${PREFER_COMPLETE:-1}"

echo "[seg-viz] host=${HOST} port=${PORT}"
echo "[seg-viz] annotation_dir=${ANNOTATION_DIR}"
echo "[seg-viz] max_samples=${MAX_SAMPLES} prefer_complete=${PREFER_COMPLETE}"
echo "[seg-viz] open: http://localhost:${PORT}/"

cd "${REPO_ROOT}"
ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --static-dir "${ROOT_DIR}"
  --annotation-dir "${ANNOTATION_DIR}"
  --max-samples "${MAX_SAMPLES}"
)

if [[ "${PREFER_COMPLETE}" == "1" ]]; then
  ARGS+=(--prefer-complete)
fi

python "${ROOT_DIR}/server.py" \
  "${ARGS[@]}"
