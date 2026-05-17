#!/usr/bin/env bash
# Refresh MCQ base with filtered NeXTVideo MCQ records.
#
# By default this replaces the MCQ base source with the filtered NeXTVideo set,
# then rebuilds multi_task/base + val and optional offline frames. Set
# RUN_NEXTVIDEO_PIPELINE=false to reuse an existing FINAL_DIRECT_JSONL.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/../../training/common/multi_task_common.sh"

TAG="${TAG:-nextvideo_reward_0p0_0p375_n3237}"
NEXTVIDEO_PIPELINE_SCRIPT="${REPO_ROOT}/video_proxy/data/base_sources/mcq/nextvideo/run_pipeline.sh"
NEXTVIDEO_OUTPUT_ROOT="${NEXTVIDEO_OUTPUT_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/rollouts/mcq_nextvideo_qwen3_vl_4b_roll8_leq3of8}"
FINAL_DIRECT_JSONL="${FINAL_DIRECT_JSONL:-${NEXTVIDEO_OUTPUT_ROOT}/train_final_direct.jsonl}"

RUN_NEXTVIDEO_PIPELINE="${RUN_NEXTVIDEO_PIPELINE:-true}"
RUN_SETUP="${RUN_SETUP:-true}"
RUN_FRAME_EXTRACTION="${RUN_FRAME_EXTRACTION:-true}"
RUN_CHECK="${RUN_CHECK:-true}"
CHECK_FRAME_JSONL="${CHECK_FRAME_JSONL:-${RUN_FRAME_EXTRACTION}}"
CHECK_FRAME_FILES="${CHECK_FRAME_FILES:-false}"
VAL_MCQ_N_EFFECTIVE="${VAL_MCQ_N:-600}"
REWARD_MIN="${REWARD_MIN:-0.0}"
REWARD_MAX="${REWARD_MAX:-0.375}"
TARGET_TOTAL="${TARGET_TOTAL:-3237}"

MCQ_REFRESH_DIR="${MCQ_REFRESH_DIR:-${REPO_ROOT}/video_proxy/data/base_sources/mcq/results/base_refresh/${TAG}}"
MCQ_BASE_SOURCE_COPY="${MCQ_REFRESH_DIR}/mcq_nextvideo_${TAG}_direct.jsonl"
MCQ_CHECK_SUMMARY="${MCQ_REFRESH_DIR}/mcq_nextvideo_${TAG}_check_summary.json"

MCQ_BASE_JSONL="${MULTI_TASK_DATA_ROOT}/base/mcq_train_filtered.jsonl"
MCQ_BASE_FRAMES_JSONL="${MULTI_TASK_DATA_ROOT}/base/mcq_train_filtered_frames.jsonl"
MCQ_VAL_JSONL="${MULTI_TASK_DATA_ROOT}/val/mcq_val_${VAL_MCQ_N_EFFECTIVE}.jsonl"
MCQ_VAL_FRAMES_JSONL="${MULTI_TASK_DATA_ROOT}/val/mcq_val_${VAL_MCQ_N_EFFECTIVE}_frames.jsonl"

echo "============================================"
echo " Refresh MCQ Base With NeXTVideo"
echo " Pipeline:       ${NEXTVIDEO_PIPELINE_SCRIPT}"
echo " Pipeline root:  ${NEXTVIDEO_OUTPUT_ROOT}"
echo " Final source:   ${FINAL_DIRECT_JSONL}"
echo " Target total:   ${TARGET_TOTAL}"
echo " Data root:      ${MULTI_TASK_DATA_ROOT}"
echo " Val MCQ N:      ${VAL_MCQ_N_EFFECTIVE}"
echo " Output dir:     ${MCQ_REFRESH_DIR}"
echo " Frames:         ${RUN_FRAME_EXTRACTION}"
echo " Check:          ${RUN_CHECK}"
echo "============================================"

mkdir -p "${MCQ_REFRESH_DIR}"

case "${RUN_NEXTVIDEO_PIPELINE}" in
    true|TRUE|1|yes|YES)
        echo ""
        echo "=== Step 1: run NeXTVideo rollout/filter pipeline ==="
        OUTPUT_ROOT="${NEXTVIDEO_OUTPUT_ROOT}" \
        TARGET_TOTAL="${TARGET_TOTAL}" \
        MIN_ACC="${REWARD_MIN}" \
        MAX_ACC="${REWARD_MAX}" \
        bash "${NEXTVIDEO_PIPELINE_SCRIPT}"
        ;;
    *)
        echo ""
        echo "=== Step 1: NeXTVideo pipeline skipped (RUN_NEXTVIDEO_PIPELINE=${RUN_NEXTVIDEO_PIPELINE}) ==="
        ;;
esac

if [[ ! -s "${FINAL_DIRECT_JSONL}" ]]; then
    echo "[refresh-nextvideo] missing final direct source: ${FINAL_DIRECT_JSONL}" >&2
    exit 1
fi

cp "${FINAL_DIRECT_JSONL}" "${MCQ_BASE_SOURCE_COPY}"

case "${RUN_SETUP}" in
    true|TRUE|1|yes|YES)
        echo ""
        echo "=== Step 2: refresh multi-task base MCQ train/val ==="
        TASKS=mcq \
        FORCE=true \
        VAL_MCQ_N="${VAL_MCQ_N_EFFECTIVE}" \
        MCQ_SOURCE="${MCQ_BASE_SOURCE_COPY}" \
        bash "${SCRIPT_DIR}/setup_base_data.sh"
        ;;
    *)
        echo ""
        echo "=== Step 2: setup skipped (RUN_SETUP=${RUN_SETUP}) ==="
        ;;
esac

case "${RUN_FRAME_EXTRACTION}" in
    true|TRUE|1|yes|YES)
        echo ""
        echo "=== Step 3: extract offline frames for current MCQ base/val ==="
        PREPARE_TG_FRAMES=false \
        PREPARE_MCQ_FRAMES=true \
        PREPARE_VAL_FRAMES=true \
        MCQ_TRAIN_INPUT="${MCQ_BASE_JSONL}" \
        MCQ_VAL_INPUT="${MCQ_VAL_JSONL}" \
        bash "${SCRIPT_DIR}/prepare_base_offline_frames.sh"
        ;;
    *)
        echo ""
        echo "=== Step 3: frame extraction skipped (RUN_FRAME_EXTRACTION=${RUN_FRAME_EXTRACTION}) ==="
        ;;
esac

case "${RUN_CHECK}" in
    true|TRUE|1|yes|YES)
        echo ""
        echo "=== Step 4: check MCQ prompt/answer and frame JSONL ==="
        CHECK_JSONL_ARGS=(--jsonl "${MCQ_BASE_JSONL}" --jsonl "${MCQ_VAL_JSONL}")
        CHECK_FRAME_ARGS=()
        if [[ "${CHECK_FRAME_JSONL,,}" == "true" ]]; then
            CHECK_FRAME_ARGS=(--frames-jsonl "${MCQ_BASE_FRAMES_JSONL}" --frames-jsonl "${MCQ_VAL_FRAMES_JSONL}")
        fi

        CHECK_EXTRA_ARGS=()
        if [[ "${CHECK_FRAME_FILES,,}" == "true" ]]; then
            CHECK_EXTRA_ARGS+=(--check-frame-files)
        fi

        python3 "${REPO_ROOT}/video_proxy/data/base_sources/mcq/check_format.py" \
            "${CHECK_JSONL_ARGS[@]}" \
            "${CHECK_FRAME_ARGS[@]}" \
            "${CHECK_EXTRA_ARGS[@]}" \
            --min-mean-reward "${REWARD_MIN}" \
            --max-mean-reward "${REWARD_MAX}" \
            --summary-json "${MCQ_CHECK_SUMMARY}"
        ;;
    *)
        echo ""
        echo "=== Step 4: check skipped (RUN_CHECK=${RUN_CHECK}) ==="
        ;;
esac

echo ""
echo "============================================"
echo " NeXTVideo MCQ base refresh done"
echo " Source copy:      ${MCQ_BASE_SOURCE_COPY}"
echo " Base MCQ train:   ${MCQ_BASE_JSONL}"
echo " Base MCQ frames:  ${MCQ_BASE_FRAMES_JSONL}"
echo " MCQ val:          ${MCQ_VAL_JSONL}"
echo " MCQ val frames:   ${MCQ_VAL_FRAMES_JSONL}"
echo " Check summary:    ${MCQ_CHECK_SUMMARY}"
echo "============================================"
