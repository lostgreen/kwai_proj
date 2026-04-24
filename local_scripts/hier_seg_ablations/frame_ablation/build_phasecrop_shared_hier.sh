#!/usr/bin/env bash
# =============================================================
# Build phase-crop hier-seg shared-frame manifests for frame ablations.
#
# This intentionally avoids the legacy full-video L2 manifest:
#   L1: full source video
#   L2: one L1 phase per sample, event times zero-based to phase start
#   L3: one L2 event clip per sample
#
# Output defaults:
#   /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/train_phasecrop/
#     train_all.jsonl
#     val_all.jsonl
#     train_all_shared_frames.jsonl
#     val_all_shared_frames.jsonl
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

HIER_DATA_ROOT="${HIER_DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1}"
ANNOTATION_DIR="${ANNOTATION_DIR:-${HIER_DATA_ROOT}/annotations}"
OUTPUT_DIR="${OUTPUT_DIR:-${HIER_DATA_ROOT}/train_phasecrop}"
FRAMES_ROOT="${FRAMES_ROOT:-${HIER_DATA_ROOT}/frame_cache/source_2fps}"

CACHE_FPS="${CACHE_FPS:-2.0}"
L1_FPS="${L1_FPS:-2}"
WORKERS="${WORKERS:-16}"
JPEG_QUALITY="${JPEG_QUALITY:-2}"
OVERWRITE_CACHE="${OVERWRITE_CACHE:-false}"
CHECK_FRAME_FILES="${CHECK_FRAME_FILES:-false}"

L1_TRAIN="${L1_TRAIN:-4000}"
L2_TRAIN="${L2_TRAIN:-8000}"
L3_TRAIN="${L3_TRAIN:-8000}"
L1_VAL="${L1_VAL:-180}"
L2_VAL="${L2_VAL:-360}"
L3_VAL="${L3_VAL:-360}"
L1_BALANCE="${L1_BALANCE:-4800}"
L2_BALANCE="${L2_BALANCE:-9600}"
L3_BALANCE="${L3_BALANCE:-9600}"

L1_MIN_PHASES="${L1_MIN_PHASES:-0}"
L1_MAX_PHASES="${L1_MAX_PHASES:-999}"
L2_MIN_EVENTS="${L2_MIN_EVENTS:-2}"
L2_MAX_EVENTS="${L2_MAX_EVENTS:-999}"
L3_MIN_ACTIONS="${L3_MIN_ACTIONS:-3}"
L3_MAX_ACTIONS="${L3_MAX_ACTIONS:-999}"
SEED="${SEED:-42}"

echo "============================================================"
echo "  Build Phase-Crop Hier Shared Frames"
echo "============================================================"
echo "  HIER_DATA_ROOT: ${HIER_DATA_ROOT}"
echo "  ANNOTATION_DIR: ${ANNOTATION_DIR}"
echo "  OUTPUT_DIR:     ${OUTPUT_DIR}"
echo "  FRAMES_ROOT:    ${FRAMES_ROOT}"
echo "  CACHE_FPS:      ${CACHE_FPS}"
echo "  L1_FPS:         ${L1_FPS}"
echo "  TRAIN:          L1=${L1_TRAIN} L2=${L2_TRAIN} L3=${L3_TRAIN}"
echo "  VAL:            L1=${L1_VAL} L2=${L2_VAL} L3=${L3_VAL}"
echo "  L2_MODE:        phase"
echo "============================================================"

mkdir -p "${OUTPUT_DIR}" "${FRAMES_ROOT}"

build_level() {
    local level="$1"
    local train_n="$2"
    local val_n="$3"
    local balance_n="$4"
    local out_dir="${OUTPUT_DIR}/${level}"

    echo ""
    echo "[phasecrop] Building ${level}: train=${train_n} val=${val_n} balance=${balance_n}"
    python3 "${REPO_ROOT}/proxy_data/youcook2_seg/hier_seg_annotation/build_hier_data.py" \
        --annotation-dir "${ANNOTATION_DIR}" \
        --output-dir "${out_dir}" \
        --levels "${level}" \
        --l1-fps "${L1_FPS}" \
        --l2-mode phase \
        --l1-min-phases "${L1_MIN_PHASES}" \
        --l1-max-phases "${L1_MAX_PHASES}" \
        --l2-min-events "${L2_MIN_EVENTS}" \
        --l2-max-events "${L2_MAX_EVENTS}" \
        --l3-min-actions "${L3_MIN_ACTIONS}" \
        --l3-max-actions "${L3_MAX_ACTIONS}" \
        --complete-only \
        --balance-per-level "${balance_n}" \
        --train-per-level "${train_n}" \
        --total-val "${val_n}" \
        --seed "${SEED}"
}

build_level L1 "${L1_TRAIN}" "${L1_VAL}" "${L1_BALANCE}"
build_level L2 "${L2_TRAIN}" "${L2_VAL}" "${L2_BALANCE}"
build_level L3_seg "${L3_TRAIN}" "${L3_VAL}" "${L3_BALANCE}"

RAW_TRAIN="${OUTPUT_DIR}/train_all.jsonl"
RAW_VAL="${OUTPUT_DIR}/val_all.jsonl"
SHARED_TRAIN="${OUTPUT_DIR}/train_all_shared_frames.jsonl"
SHARED_VAL="${OUTPUT_DIR}/val_all_shared_frames.jsonl"

echo ""
echo "[phasecrop] Merging raw manifests..."
: > "${RAW_TRAIN}"
: > "${RAW_VAL}"
for level in L1 L2 L3_seg; do
    cat "${OUTPUT_DIR}/${level}/train.jsonl" >> "${RAW_TRAIN}"
    cat "${OUTPUT_DIR}/${level}/val.jsonl" >> "${RAW_VAL}"
done
echo "  train: $(wc -l < "${RAW_TRAIN}") -> ${RAW_TRAIN}"
echo "  val:   $(wc -l < "${RAW_VAL}") -> ${RAW_VAL}"

rewrite_shared() {
    local input_jsonl="$1"
    local output_jsonl="$2"
    local overwrite_flag=()
    if [[ "${OVERWRITE_CACHE,,}" =~ ^(true|1|yes)$ ]]; then
        overwrite_flag=(--overwrite-cache)
    fi

    echo ""
    echo "[phasecrop] Rewriting shared frames: ${input_jsonl} -> ${output_jsonl}"
    python3 "${REPO_ROOT}/local_scripts/data/rewrite_hier_to_shared_frames.py" \
        --input-jsonl "${input_jsonl}" \
        --output-jsonl "${output_jsonl}" \
        --annotation-dir "${ANNOTATION_DIR}" \
        --frames-root "${FRAMES_ROOT}" \
        --cache-fps "${CACHE_FPS}" \
        --l1-view-fps 2.0 \
        --l2-full-view-fps 2.0 \
        --default-view-fps 2.0 \
        --jpeg-quality "${JPEG_QUALITY}" \
        --workers "${WORKERS}" \
        "${overwrite_flag[@]}"
}

rewrite_shared "${RAW_TRAIN}" "${SHARED_TRAIN}"
rewrite_shared "${RAW_VAL}" "${SHARED_VAL}"

CHECK_FRAME_FILES_FLAG=()
if [[ "${CHECK_FRAME_FILES,,}" =~ ^(true|1|yes)$ ]]; then
    CHECK_FRAME_FILES_FLAG=(--check-frame-files)
fi

echo ""
echo "[phasecrop] Validating phase-crop shared manifests..."
python3 "${REPO_ROOT}/local_scripts/data/check_hier_phasecrop_manifest.py" \
    --jsonl "${SHARED_TRAIN}" \
    --jsonl "${SHARED_VAL}" \
    "${CHECK_FRAME_FILES_FLAG[@]}"

echo ""
echo "============================================================"
echo "  Phase-crop hier shared manifests ready"
echo "  HIER_TRAIN=${SHARED_TRAIN}"
echo "  HIER_VAL_SOURCE=${SHARED_VAL}"
echo "============================================================"
