#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/multi_task_common.sh"

LLAVA_ROLLOUT_DIR="${LLAVA_ROLLOUT_DIR:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/results_qwen3_vl_8b_roll8_leq3of8}"
LLAVA_MCQ_INPUT="${LLAVA_MCQ_INPUT:-${LLAVA_ROLLOUT_DIR}/mcq_all.jsonl}"
LLAVA_REPORT_GLOB="${LLAVA_REPORT_GLOB:-${LLAVA_ROLLOUT_DIR}/_shard*_report.jsonl}"
LLAVA_REPORT_JSONL="${LLAVA_REPORT_JSONL:-}"
MCQ_BASE_SOURCE="${MCQ_BASE_SOURCE:-${REPO_ROOT}/proxy_data/llava_video_178k/results/train_final_direct.jsonl}"

REWARD_MIN="${REWARD_MIN:-0.0}"
REWARD_MAX="${REWARD_MAX:-0.375}"
VAL_MCQ_N_EFFECTIVE="${VAL_MCQ_N:-600}"
SEED="${SEED:-42}"
TARGET_TOTAL="${TARGET_TOTAL:-0}"
RUN_SETUP="${RUN_SETUP:-true}"
RUN_FRAME_EXTRACTION="${RUN_FRAME_EXTRACTION:-true}"
RUN_CHECK="${RUN_CHECK:-true}"
CHECK_FRAME_JSONL="${CHECK_FRAME_JSONL:-${RUN_FRAME_EXTRACTION}}"
CHECK_FRAME_FILES="${CHECK_FRAME_FILES:-false}"
SKIP_BAD_REPORT_LINES="${SKIP_BAD_REPORT_LINES:-true}"

TAG_DEFAULT="reward_${REWARD_MIN//./p}_${REWARD_MAX//./p}"
TAG="${TAG:-${TAG_DEFAULT}}"
MCQ_REFRESH_DIR="${MCQ_REFRESH_DIR:-${REPO_ROOT}/proxy_data/llava_video_178k/results/base_refresh/${TAG}}"
MCQ_SELECTED="${MCQ_REFRESH_DIR}/mcq_llava_${TAG}_direct.jsonl"
MCQ_MERGED="${MCQ_REFRESH_DIR}/mcq_base_plus_llava_${TAG}_direct.jsonl"
MCQ_SELECTION_SUMMARY="${MCQ_REFRESH_DIR}/mcq_llava_${TAG}_summary.json"
MCQ_CHECK_SUMMARY="${MCQ_REFRESH_DIR}/mcq_base_${TAG}_check_summary.json"

MCQ_BASE_JSONL="${MULTI_TASK_DATA_ROOT}/base/mcq_train_filtered.jsonl"
MCQ_BASE_FRAMES_JSONL="${MULTI_TASK_DATA_ROOT}/base/mcq_train_filtered_frames.jsonl"
MCQ_VAL_JSONL="${MULTI_TASK_DATA_ROOT}/val/mcq_val_${VAL_MCQ_N_EFFECTIVE}.jsonl"
MCQ_VAL_FRAMES_JSONL="${MULTI_TASK_DATA_ROOT}/val/mcq_val_${VAL_MCQ_N_EFFECTIVE}_frames.jsonl"

echo "============================================"
echo " Refresh MCQ Base With LLaVA Shards"
echo " Rollout dir:      ${LLAVA_ROLLOUT_DIR}"
echo " MCQ input:        ${LLAVA_MCQ_INPUT}"
echo " Report glob:      ${LLAVA_REPORT_GLOB}"
echo " Report JSONL:     ${LLAVA_REPORT_JSONL:-<none>}"
echo " Existing source:  ${MCQ_BASE_SOURCE}"
echo " Reward range:     [${REWARD_MIN}, ${REWARD_MAX}]"
echo " Target total:     ${TARGET_TOTAL} (0 = keep all)"
echo " Val MCQ N:        ${VAL_MCQ_N_EFFECTIVE}"
echo " Output dir:       ${MCQ_REFRESH_DIR}"
echo " Frames:           ${RUN_FRAME_EXTRACTION}"
echo " Check:            ${RUN_CHECK}"
echo "============================================"

mkdir -p "${MCQ_REFRESH_DIR}"

echo ""
echo "=== Step 1: select low-reward shard records and merge source ==="
SELECT_ARGS=(
    --input "${LLAVA_MCQ_INPUT}"
    --report-glob "${LLAVA_REPORT_GLOB}"
    --output "${MCQ_SELECTED}"
    --base-jsonl "${MCQ_BASE_SOURCE}"
    --merged-output "${MCQ_MERGED}"
    --summary-json "${MCQ_SELECTION_SUMMARY}"
    --min-mean-reward "${REWARD_MIN}"
    --max-mean-reward "${REWARD_MAX}"
    --target-total "${TARGET_TOTAL}"
    --seed "${SEED}"
)

if [[ -n "${LLAVA_REPORT_JSONL}" ]]; then
    SELECT_ARGS+=(--report "${LLAVA_REPORT_JSONL}")
fi
if [[ "${SKIP_BAD_REPORT_LINES,,}" == "true" ]]; then
    SELECT_ARGS+=(--skip-bad-report-lines)
fi

python3 "${REPO_ROOT}/proxy_data/llava_video_178k/select_mcq_from_rollout_shards.py" "${SELECT_ARGS[@]}"

case "${RUN_SETUP}" in
    true|TRUE|1|yes|YES)
        echo ""
        echo "=== Step 2: refresh multi-task base MCQ train/val ==="
        TASKS=mcq \
        FORCE=true \
        VAL_MCQ_N="${VAL_MCQ_N_EFFECTIVE}" \
        MCQ_SOURCE="${MCQ_MERGED}" \
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

        python3 "${REPO_ROOT}/proxy_data/llava_video_178k/check_mcq_prompt_format.py" \
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
echo " MCQ base refresh done"
echo " Selected shard MCQ: ${MCQ_SELECTED}"
echo " Merged MCQ source:  ${MCQ_MERGED}"
echo " Selection summary:  ${MCQ_SELECTION_SUMMARY}"
echo " Base MCQ train:     ${MCQ_BASE_JSONL}"
echo " Base MCQ frames:    ${MCQ_BASE_FRAMES_JSONL}"
echo " MCQ val:            ${MCQ_VAL_JSONL}"
echo " MCQ val frames:     ${MCQ_VAL_FRAMES_JSONL}"
echo " Check summary:      ${MCQ_CHECK_SUMMARY}"
echo "============================================"
