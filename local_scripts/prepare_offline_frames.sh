#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/multi_task_common.sh"

EXP_NAME="${EXP_NAME:-multi_task_demo_8gpu}"
EXP_DATA_DIR="${EXPERIMENTS_DIR}/${EXP_NAME}"

TRAIN_INPUT_JSONL="${TRAIN_INPUT_JSONL:-${EXP_DATA_DIR}/train.jsonl}"
VAL_INPUT_JSONL="${VAL_INPUT_JSONL:-${EXP_DATA_DIR}/val.jsonl}"

OFFLINE_FRAME_ROOT="${OFFLINE_FRAME_ROOT:-${MULTI_TASK_DATA_ROOT}/offline_frames}"
OFFLINE_FRAME_CACHE_ROOT="${OFFLINE_FRAME_CACHE_ROOT:-${OFFLINE_FRAME_ROOT}/cache}"
TRAIN_FRAMES_ROOT="${TRAIN_FRAMES_ROOT:-${OFFLINE_FRAME_CACHE_ROOT}}"
VAL_FRAMES_ROOT="${VAL_FRAMES_ROOT:-${OFFLINE_FRAME_CACHE_ROOT}}"

TRAIN_OUTPUT_JSONL="${TRAIN_OUTPUT_JSONL:-${EXP_DATA_DIR}/train_frames.jsonl}"
VAL_OUTPUT_JSONL="${VAL_OUTPUT_JSONL:-${EXP_DATA_DIR}/val_frames.jsonl}"

OFFLINE_FRAME_WORKERS="${OFFLINE_FRAME_WORKERS:-16}"
OFFLINE_FRAME_JPEG_QUALITY="${OFFLINE_FRAME_JPEG_QUALITY:-2}"
OFFLINE_FRAME_ABSOLUTE_PATHS="${OFFLINE_FRAME_ABSOLUTE_PATHS:-true}"
INPUT_IMAGE_DIR="${INPUT_IMAGE_DIR:-}"
OVERWRITE="${OVERWRITE:-false}"

if [[ "${OFFLINE_FRAME_ABSOLUTE_PATHS,,}" != "true" ]]; then
    echo "[offline-frames] This wrapper currently requires OFFLINE_FRAME_ABSOLUTE_PATHS=true" >&2
    echo "[offline-frames] because train/val frames are written to different roots." >&2
    exit 1
fi

if [[ ! -f "${TRAIN_INPUT_JSONL}" ]]; then
    echo "[offline-frames] train input not found: ${TRAIN_INPUT_JSONL}" >&2
    exit 1
fi

if [[ ! -f "${VAL_INPUT_JSONL}" ]]; then
    echo "[offline-frames] val input not found: ${VAL_INPUT_JSONL}" >&2
    exit 1
fi

mkdir -p "${OFFLINE_FRAME_ROOT}"

COMMON_ARGS=(
    --target-fps "${VIDEO_FPS}"
    --fallback-fps "1.0"
    --max-frames "${MAX_FRAMES}"
    --workers "${OFFLINE_FRAME_WORKERS}"
    --jpeg-quality "${OFFLINE_FRAME_JPEG_QUALITY}"
)

if [[ -n "${INPUT_IMAGE_DIR}" ]]; then
    COMMON_ARGS+=(--input-image-dir "${INPUT_IMAGE_DIR}")
fi

if [[ "${OFFLINE_FRAME_ABSOLUTE_PATHS,,}" == "true" ]]; then
    COMMON_ARGS+=(--absolute-paths)
fi

if [[ "${OVERWRITE,,}" == "true" ]]; then
    COMMON_ARGS+=(--overwrite)
fi

echo "[offline-frames] EXP_NAME=${EXP_NAME}"
echo "[offline-frames] train input: ${TRAIN_INPUT_JSONL}"
echo "[offline-frames] val input:   ${VAL_INPUT_JSONL}"
echo "[offline-frames] frame root:   ${OFFLINE_FRAME_ROOT}"
echo "[offline-frames] train frames: ${TRAIN_FRAMES_ROOT}"
echo "[offline-frames] val frames:   ${VAL_FRAMES_ROOT}"
echo "[offline-frames] train output: ${TRAIN_OUTPUT_JSONL}"
echo "[offline-frames] val output:   ${VAL_OUTPUT_JSONL}"

python3 "${REPO_ROOT}/local_scripts/data/rewrite_videos_to_frames.py" \
    --input-jsonl "${TRAIN_INPUT_JSONL}" \
    --output-jsonl "${TRAIN_OUTPUT_JSONL}" \
    --frames-root "${TRAIN_FRAMES_ROOT}" \
    "${COMMON_ARGS[@]}"

python3 "${REPO_ROOT}/local_scripts/data/rewrite_videos_to_frames.py" \
    --input-jsonl "${VAL_INPUT_JSONL}" \
    --output-jsonl "${VAL_OUTPUT_JSONL}" \
    --frames-root "${VAL_FRAMES_ROOT}" \
    "${COMMON_ARGS[@]}"

echo "[offline-frames] done"
echo "[offline-frames] training can use:"
echo "  TRAIN_FILE=${TRAIN_OUTPUT_JSONL}"
echo "  TEST_FILE=${VAL_OUTPUT_JSONL}"
