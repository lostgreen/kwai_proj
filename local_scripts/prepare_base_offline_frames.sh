#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/multi_task_common.sh"

BASE_DIR="${MULTI_TASK_DATA_ROOT}/base"
VAL_DIR="${MULTI_TASK_DATA_ROOT}/val"
BASE_FRAME_CACHE_ROOT="${BASE_FRAME_CACHE_ROOT:-${MULTI_TASK_DATA_ROOT}/offline_frames/base_cache}"

BASE_FRAME_TARGET_FPS="${BASE_FRAME_TARGET_FPS:-${VIDEO_FPS}}"
BASE_FRAME_FALLBACK_FPS="${BASE_FRAME_FALLBACK_FPS:-1.0}"
BASE_FRAME_MAX_FRAMES="${BASE_FRAME_MAX_FRAMES:-${MAX_FRAMES}}"
BASE_FRAME_MIN_FPS="${BASE_FRAME_MIN_FPS:-0.25}"
BASE_FRAME_WORKERS="${BASE_FRAME_WORKERS:-16}"
BASE_FRAME_JPEG_QUALITY="${BASE_FRAME_JPEG_QUALITY:-2}"
BASE_FRAME_OVERWRITE="${BASE_FRAME_OVERWRITE:-false}"

TG_TRAIN_INPUT="${TG_TRAIN_INPUT:-${BASE_DIR}/tg_train.jsonl}"
MCQ_TRAIN_INPUT="${MCQ_TRAIN_INPUT:-${BASE_DIR}/mcq_train_filtered.jsonl}"

mkdir -p "${BASE_FRAME_CACHE_ROOT}"

COMMON_ARGS=(
    --target-fps "${BASE_FRAME_TARGET_FPS}"
    --fallback-fps "${BASE_FRAME_FALLBACK_FPS}"
    --max-frames "${BASE_FRAME_MAX_FRAMES}"
    --min-fps "${BASE_FRAME_MIN_FPS}"
    --workers "${BASE_FRAME_WORKERS}"
    --jpeg-quality "${BASE_FRAME_JPEG_QUALITY}"
    --absolute-paths
)

if [[ "${BASE_FRAME_OVERWRITE,,}" == "true" ]]; then
    COMMON_ARGS+=(--overwrite)
fi

rewrite_one() {
    local input_jsonl="$1"
    local output_jsonl="$2"
    local label="$3"

    if [[ ! -f "${input_jsonl}" ]]; then
        echo "[base-offline-frames] skip ${label}: not found -> ${input_jsonl}"
        return 0
    fi

    echo "[base-offline-frames] ${label}"
    echo "  input : ${input_jsonl}"
    echo "  output: ${output_jsonl}"
    python3 "${REPO_ROOT}/local_scripts/data/rewrite_videos_to_frames.py" \
        --input-jsonl "${input_jsonl}" \
        --output-jsonl "${output_jsonl}" \
        --frames-root "${BASE_FRAME_CACHE_ROOT}" \
        "${COMMON_ARGS[@]}"
}

echo "============================================"
echo "  Prepare Base Offline Frames"
echo "============================================"
echo "  Data root:      ${MULTI_TASK_DATA_ROOT}"
echo "  Base dir:       ${BASE_DIR}"
echo "  Val dir:        ${VAL_DIR}"
echo "  Frame cache:    ${BASE_FRAME_CACHE_ROOT}"
echo "  target_fps:     ${BASE_FRAME_TARGET_FPS}"
echo "  fallback_fps:   ${BASE_FRAME_FALLBACK_FPS}"
echo "  max_frames:     ${BASE_FRAME_MAX_FRAMES}"
echo "  workers:        ${BASE_FRAME_WORKERS}"
echo "  overwrite:      ${BASE_FRAME_OVERWRITE}"
echo "============================================"

rewrite_one "${TG_TRAIN_INPUT}" "${TG_TRAIN_INPUT%.jsonl}_frames.jsonl" "TG train"
rewrite_one "${MCQ_TRAIN_INPUT}" "${MCQ_TRAIN_INPUT%.jsonl}_frames.jsonl" "MCQ train"

shopt -s nullglob
for input_jsonl in "${VAL_DIR}"/tg_val_*.jsonl; do
    [[ "${input_jsonl}" == *_frames.jsonl ]] && continue
    rewrite_one "${input_jsonl}" "${input_jsonl%.jsonl}_frames.jsonl" "TG val"
done

for input_jsonl in "${VAL_DIR}"/mcq_val_*.jsonl; do
    [[ "${input_jsonl}" == *_frames.jsonl ]] && continue
    rewrite_one "${input_jsonl}" "${input_jsonl%.jsonl}_frames.jsonl" "MCQ val"
done
shopt -u nullglob

echo "[base-offline-frames] done"
echo "[base-offline-frames] tg/mcq loaders will auto-prefer *_frames.jsonl when present"
