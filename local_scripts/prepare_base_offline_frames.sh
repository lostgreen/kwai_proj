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
PREPARE_TG_FRAMES="${PREPARE_TG_FRAMES:-true}"
PREPARE_MCQ_FRAMES="${PREPARE_MCQ_FRAMES:-true}"
PREPARE_VAL_FRAMES="${PREPARE_VAL_FRAMES:-true}"

TG_TRAIN_INPUT="${TG_TRAIN_INPUT:-${BASE_DIR}/tg_train.jsonl}"
MCQ_TRAIN_INPUT="${MCQ_TRAIN_INPUT:-${BASE_DIR}/mcq_train_filtered.jsonl}"
TG_VAL_INPUT="${TG_VAL_INPUT:-}"
MCQ_VAL_INPUT="${MCQ_VAL_INPUT:-}"

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
echo "  prepare_tg:     ${PREPARE_TG_FRAMES}"
echo "  prepare_mcq:    ${PREPARE_MCQ_FRAMES}"
echo "  prepare_val:    ${PREPARE_VAL_FRAMES}"
echo "  tg_val_input:   ${TG_VAL_INPUT:-<all tg_val_*.jsonl>}"
echo "  mcq_val_input:  ${MCQ_VAL_INPUT:-<all mcq_val_*.jsonl>}"
echo "============================================"

if [[ "${PREPARE_TG_FRAMES,,}" == "true" ]]; then
    rewrite_one "${TG_TRAIN_INPUT}" "${TG_TRAIN_INPUT%.jsonl}_frames.jsonl" "TG train"
else
    echo "[base-offline-frames] skip TG train: PREPARE_TG_FRAMES=${PREPARE_TG_FRAMES}"
fi

if [[ "${PREPARE_MCQ_FRAMES,,}" == "true" ]]; then
    rewrite_one "${MCQ_TRAIN_INPUT}" "${MCQ_TRAIN_INPUT%.jsonl}_frames.jsonl" "MCQ train"
else
    echo "[base-offline-frames] skip MCQ train: PREPARE_MCQ_FRAMES=${PREPARE_MCQ_FRAMES}"
fi

if [[ "${PREPARE_VAL_FRAMES,,}" == "true" ]]; then
    shopt -s nullglob
    if [[ "${PREPARE_TG_FRAMES,,}" == "true" ]]; then
        if [[ -n "${TG_VAL_INPUT}" ]]; then
            rewrite_one "${TG_VAL_INPUT}" "${TG_VAL_INPUT%.jsonl}_frames.jsonl" "TG val"
        else
            for input_jsonl in "${VAL_DIR}"/tg_val_*.jsonl; do
                [[ "${input_jsonl}" == *_frames.jsonl ]] && continue
                rewrite_one "${input_jsonl}" "${input_jsonl%.jsonl}_frames.jsonl" "TG val"
            done
        fi
    fi

    if [[ "${PREPARE_MCQ_FRAMES,,}" == "true" ]]; then
        if [[ -n "${MCQ_VAL_INPUT}" ]]; then
            rewrite_one "${MCQ_VAL_INPUT}" "${MCQ_VAL_INPUT%.jsonl}_frames.jsonl" "MCQ val"
        else
            for input_jsonl in "${VAL_DIR}"/mcq_val_*.jsonl; do
                [[ "${input_jsonl}" == *_frames.jsonl ]] && continue
                rewrite_one "${input_jsonl}" "${input_jsonl%.jsonl}_frames.jsonl" "MCQ val"
            done
        fi
    fi
    shopt -u nullglob
else
    echo "[base-offline-frames] skip val frames: PREPARE_VAL_FRAMES=${PREPARE_VAL_FRAMES}"
fi

echo "[base-offline-frames] done"
echo "[base-offline-frames] tg/mcq loaders will auto-prefer *_frames.jsonl when present"
