#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/multi_task_common.sh"

HIER_INPUT_JSONL="${HIER_INPUT_JSONL:-${HIER_TRAIN}}"
HIER_ANNOTATION_DIR="${HIER_ANNOTATION_DIR:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/annotations_reclassified}"
SHARED_FRAME_ROOT="${SHARED_FRAME_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/frame_cache/source_2fps}"
HIER_OUTPUT_JSONL="${HIER_OUTPUT_JSONL:-${HIER_INPUT_JSONL%.jsonl}_shared_frames.jsonl}"

CACHE_FPS="${CACHE_FPS:-2.0}"
L1_VIEW_FPS="${L1_VIEW_FPS:-1.0}"
L2_FULL_VIEW_FPS="${L2_FULL_VIEW_FPS:-1.0}"
DEFAULT_VIEW_FPS="${DEFAULT_VIEW_FPS:-2.0}"
SHARED_FRAME_WORKERS="${SHARED_FRAME_WORKERS:-16}"
SHARED_FRAME_JPEG_QUALITY="${SHARED_FRAME_JPEG_QUALITY:-2}"
OVERWRITE_CACHE="${OVERWRITE_CACHE:-false}"

if [[ ! -f "${HIER_INPUT_JSONL}" ]]; then
  echo "[shared-frames] input jsonl not found: ${HIER_INPUT_JSONL}" >&2
  exit 1
fi

if [[ ! -d "${HIER_ANNOTATION_DIR}" ]]; then
  echo "[shared-frames] annotation dir not found: ${HIER_ANNOTATION_DIR}" >&2
  exit 1
fi

ARGS=(
  --input-jsonl "${HIER_INPUT_JSONL}"
  --output-jsonl "${HIER_OUTPUT_JSONL}"
  --annotation-dir "${HIER_ANNOTATION_DIR}"
  --frames-root "${SHARED_FRAME_ROOT}"
  --cache-fps "${CACHE_FPS}"
  --l1-view-fps "${L1_VIEW_FPS}"
  --l2-full-view-fps "${L2_FULL_VIEW_FPS}"
  --default-view-fps "${DEFAULT_VIEW_FPS}"
  --jpeg-quality "${SHARED_FRAME_JPEG_QUALITY}"
  --workers "${SHARED_FRAME_WORKERS}"
)

if [[ "${OVERWRITE_CACHE,,}" == "true" ]]; then
  ARGS+=(--overwrite-cache)
fi

BUILD_ARGS=(
  --annotation-dir "${HIER_ANNOTATION_DIR}"
  --frames-root "${SHARED_FRAME_ROOT}"
  --cache-fps "${CACHE_FPS}"
  --jpeg-quality "${SHARED_FRAME_JPEG_QUALITY}"
  --workers "${SHARED_FRAME_WORKERS}"
)
if [[ "${OVERWRITE_CACHE,,}" == "true" ]]; then
  BUILD_ARGS+=(--overwrite)
fi

echo "[shared-frames] hier input:    ${HIER_INPUT_JSONL}"
echo "[shared-frames] annotations:   ${HIER_ANNOTATION_DIR}"
echo "[shared-frames] cache root:    ${SHARED_FRAME_ROOT}"
echo "[shared-frames] output jsonl:  ${HIER_OUTPUT_JSONL}"
echo "[shared-frames] cache fps:     ${CACHE_FPS}"
echo "[shared-frames] view fps:      L1=${L1_VIEW_FPS} L2full=${L2_FULL_VIEW_FPS} default=${DEFAULT_VIEW_FPS}"

python3 "${REPO_ROOT}/local_scripts/data/build_source_frame_cache.py" "${BUILD_ARGS[@]}"

python3 "${REPO_ROOT}/local_scripts/data/rewrite_hier_to_shared_frames.py" "${ARGS[@]}"
