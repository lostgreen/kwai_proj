#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8890}"
ANNOTATION_DIR="${ANNOTATION_DIR:-proxy_data/youcook2_seg_annotation/annotations}"

echo "[seg-viz] host=${HOST} port=${PORT}"
echo "[seg-viz] annotation_dir=${ANNOTATION_DIR}"
echo "[seg-viz] open: http://localhost:${PORT}/"

cd "${REPO_ROOT}"
python "${ROOT_DIR}/server.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --static-dir "${ROOT_DIR}" \
  --annotation-dir "${ANNOTATION_DIR}"
