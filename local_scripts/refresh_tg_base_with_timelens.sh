#!/usr/bin/env bash
# Refresh temporal-grounding base data with selected TimeLens short-video TG.
#
# Flow:
#   1. Ensure TimeLens rollout analysis exists.
#   2. Select TimeLens query samples with mean IoU in [IOU_MIN, IOU_MAX].
#   3. Rewrite TimeRFT train and TGBench val prompts into TG-Bench style.
#   4. Merge TimeRFT + selected TimeLens into a new TG train source.
#   5. Refresh multi-task base TG train/val and optionally rewrite TG videos to frames.
#   6. Check prompt/answer format, TimeLens IoU bounds, and frame-list structure.
#
# Usage from train/:
#   bash local_scripts/refresh_tg_base_with_timelens.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/multi_task_common.sh"

IOU_MIN="${IOU_MIN:-0.1}"
IOU_MAX="${IOU_MAX:-0.4}"
TAG="${TAG:-iou0p1_0p4}"
RUN_FRAME_EXTRACTION="${RUN_FRAME_EXTRACTION:-true}"
RUN_CHECK="${RUN_CHECK:-true}"
CHECK_FRAME_JSONL="${CHECK_FRAME_JSONL:-${RUN_FRAME_EXTRACTION}}"
CHECK_FRAME_FILES="${CHECK_FRAME_FILES:-false}"
VAL_TG_N_EFFECTIVE="${VAL_TG_N:-150}"

TIMELENS_ROLLOUT_DIR="${TIMELENS_ROLLOUT_DIR:-${REPO_ROOT}/proxy_data/data_curation/results/timelens_100k_short/tg_rollout_qwen3_vl_8b_roll8}"
TIMELENS_TG_INPUT="${TIMELENS_TG_INPUT:-${TIMELENS_ROLLOUT_DIR}/tg_rollout_input.jsonl}"
TIMELENS_ANALYSIS_DIR="${TIMELENS_ANALYSIS_DIR:-${TIMELENS_ROLLOUT_DIR}/analysis}"
TIMELENS_QUERY_STATS="${TIMELENS_QUERY_STATS:-${TIMELENS_ANALYSIS_DIR}/query_stats.jsonl}"

TIMERFT_SOURCE="${TIMERFT_SOURCE:-${REPO_ROOT}/proxy_data/temporal_grounding/data/tg_timerft_max256s_validated.jsonl}"
TVGBENCH_SOURCE="${TVGBENCH_SOURCE:-${REPO_ROOT}/proxy_data/temporal_grounding/data/tg_tvgbench_max256s_validated.jsonl}"
TG_REFRESH_DIR="${TG_REFRESH_DIR:-${REPO_ROOT}/proxy_data/temporal_grounding/data/timelens_refresh_${TAG}}"

SELECT_SCRIPT="${REPO_ROOT}/proxy_data/data_curation/timelens_100k/select_tg_iou_range.py"
ANALYZE_SCRIPT="${REPO_ROOT}/proxy_data/data_curation/timelens_100k/analyze_tg_rollout.py"
REWRITE_SCRIPT="${REPO_ROOT}/proxy_data/temporal_grounding/rewrite_tg_prompt_format.py"
MERGE_SCRIPT="${REPO_ROOT}/proxy_data/temporal_grounding/merge_tg_train_with_timelens.py"
CHECK_SCRIPT="${REPO_ROOT}/proxy_data/temporal_grounding/check_tg_prompt_format.py"

TIMELENS_SELECTED="${TG_REFRESH_DIR}/tg_timelens_${TAG}.jsonl"
TIMERFT_REPROMPT="${TG_REFRESH_DIR}/tg_timerft_max256s_validated_reprompt.jsonl"
TVGBENCH_REPROMPT="${TG_REFRESH_DIR}/tg_tvgbench_max256s_validated_reprompt.jsonl"
TG_MERGED="${TG_REFRESH_DIR}/tg_timerft_plus_timelens_${TAG}.jsonl"
TG_BASE_JSONL="${MULTI_TASK_DATA_ROOT}/base/tg_train.jsonl"
TG_BASE_FRAMES_JSONL="${MULTI_TASK_DATA_ROOT}/base/tg_train_frames.jsonl"
TG_VAL_JSONL="${MULTI_TASK_DATA_ROOT}/val/tg_val_${VAL_TG_N_EFFECTIVE}.jsonl"
TG_VAL_FRAMES_JSONL="${MULTI_TASK_DATA_ROOT}/val/tg_val_${VAL_TG_N_EFFECTIVE}_frames.jsonl"

echo "============================================"
echo " Refresh TG Base With TimeLens"
echo "============================================"
echo " Data root:        ${MULTI_TASK_DATA_ROOT}"
echo " TimeLens rollout: ${TIMELENS_ROLLOUT_DIR}"
echo " TimeLens input:   ${TIMELENS_TG_INPUT}"
echo " Query stats:      ${TIMELENS_QUERY_STATS}"
echo " TimeRFT source:   ${TIMERFT_SOURCE}"
echo " TGBench source:   ${TVGBENCH_SOURCE}"
echo " Val TG N:         ${VAL_TG_N_EFFECTIVE}"
echo " IoU range:        [${IOU_MIN}, ${IOU_MAX}]"
echo " Output dir:       ${TG_REFRESH_DIR}"
echo " Frames:           ${RUN_FRAME_EXTRACTION}"
echo " Check:            ${RUN_CHECK}"
echo "============================================"

mkdir -p "${TG_REFRESH_DIR}" "${TIMELENS_ANALYSIS_DIR}"

if [[ ! -f "${TIMELENS_TG_INPUT}" ]]; then
    echo "[refresh-tg] TimeLens TG input not found: ${TIMELENS_TG_INPUT}" >&2
    exit 1
fi

if [[ ! -f "${TIMELENS_QUERY_STATS}" ]]; then
    echo "[refresh-tg] query_stats missing; running analysis from reports..."
    REPORT_ARGS=()
    if [[ -s "${TIMELENS_ROLLOUT_DIR}/rollout_report.jsonl" ]]; then
        REPORT_ARGS=(--report "${TIMELENS_ROLLOUT_DIR}/rollout_report.jsonl")
    else
        shopt -s nullglob
        SHARD_REPORTS=("${TIMELENS_ROLLOUT_DIR}"/_shard*_report.jsonl)
        shopt -u nullglob
        if [[ "${#SHARD_REPORTS[@]}" -eq 0 ]]; then
            echo "[refresh-tg] no rollout_report.jsonl or _shard*_report.jsonl found in ${TIMELENS_ROLLOUT_DIR}" >&2
            exit 1
        fi
        REPORT_ARGS=(--report "${SHARD_REPORTS[@]}")
    fi
    python3 "${ANALYZE_SCRIPT}" \
        "${REPORT_ARGS[@]}" \
        --input-jsonl "${TIMELENS_TG_INPUT}" \
        --output-dir "${TIMELENS_ANALYSIS_DIR}"
fi

echo ""
echo "=== Step 1: select TimeLens TG by IoU range ==="
python3 "${SELECT_SCRIPT}" \
    --input-jsonl "${TIMELENS_TG_INPUT}" \
    --query-stats "${TIMELENS_QUERY_STATS}" \
    --output-jsonl "${TIMELENS_SELECTED}" \
    --min-iou "${IOU_MIN}" \
    --max-iou "${IOU_MAX}"

echo ""
echo "=== Step 2: rewrite TimeRFT train prompt format ==="
python3 "${REWRITE_SCRIPT}" \
    --input "${TIMERFT_SOURCE}" \
    --output "${TIMERFT_REPROMPT}"

echo ""
echo "=== Step 3: rewrite TGBench val prompt format ==="
python3 "${REWRITE_SCRIPT}" \
    --input "${TVGBENCH_SOURCE}" \
    --output "${TVGBENCH_REPROMPT}"

echo ""
echo "=== Step 4: merge TimeRFT + TimeLens TG train ==="
python3 "${MERGE_SCRIPT}" \
    --base "${TIMERFT_REPROMPT}" \
    --timelens "${TIMELENS_SELECTED}" \
    --output "${TG_MERGED}"

echo ""
echo "=== Step 5: refresh multi-task base TG train/val ==="
TASKS=tg \
FORCE=true \
VAL_TG_N="${VAL_TG_N_EFFECTIVE}" \
TG_TRAIN_SOURCE="${TG_MERGED}" \
TG_TVGBENCH_SOURCE="${TVGBENCH_REPROMPT}" \
bash "${SCRIPT_DIR}/setup_base_data.sh"

case "${RUN_FRAME_EXTRACTION}" in
    true|TRUE|1|yes|YES)
        echo ""
        echo "=== Step 6: rewrite TG base/val videos to offline frames ==="
        PREPARE_TG_FRAMES=true \
        PREPARE_MCQ_FRAMES=false \
        PREPARE_VAL_FRAMES=true \
        TG_VAL_INPUT="${TG_VAL_JSONL}" \
        bash "${SCRIPT_DIR}/prepare_base_offline_frames.sh"
        ;;
    *)
        echo ""
        echo "=== Step 6: frame extraction disabled ==="
        ;;
esac

case "${RUN_CHECK}" in
    true|TRUE|1|yes|YES)
        echo ""
        echo "=== Step 7: check TG prompt/answer and frame JSONL ==="
        CHECK_JSONL_ARGS=(--jsonl "${TG_BASE_JSONL}" --jsonl "${TG_VAL_JSONL}")
        CHECK_FRAME_ARGS=()
        if [[ "${CHECK_FRAME_JSONL,,}" == "true" ]]; then
            CHECK_FRAME_ARGS=(--frames-jsonl "${TG_BASE_FRAMES_JSONL}" --frames-jsonl "${TG_VAL_FRAMES_JSONL}")
        fi

        CHECK_EXTRA_ARGS=()
        if [[ "${CHECK_FRAME_FILES,,}" == "true" ]]; then
            CHECK_EXTRA_ARGS+=(--check-frame-files)
        fi

        python3 "${CHECK_SCRIPT}" \
            "${CHECK_JSONL_ARGS[@]}" \
            "${CHECK_FRAME_ARGS[@]}" \
            --timelens-min-iou "${IOU_MIN}" \
            --timelens-max-iou "${IOU_MAX}" \
            --summary-json "${TG_REFRESH_DIR}/tg_base_format_check.json" \
            "${CHECK_EXTRA_ARGS[@]}"
        ;;
    *)
        echo ""
        echo "=== Step 7: check disabled ==="
        ;;
esac

echo ""
echo "============================================"
echo " TG base refresh done"
echo " Merged TG source: ${TG_MERGED}"
echo " Base TG train:    ${TG_BASE_JSONL}"
echo " Base TG frames:   ${TG_BASE_FRAMES_JSONL}"
echo " TG val:           ${TG_VAL_JSONL}"
echo " TG val frames:    ${TG_VAL_FRAMES_JSONL}"
echo "============================================"
