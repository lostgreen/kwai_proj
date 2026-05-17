#!/usr/bin/env bash
# NeXTVideo MCQ rollout pipeline.
#
# Flow:
#   1. Convert NeXTVideo train.jsonl to shared MCQ JSONL.
#   2. Run Qwen3-VL offline rollout with the existing MCQ reward.
#   3. Select low-reward records from the rollout report and downsample to a target count.
#
# Usage from train/:
#   bash video_proxy/data/base_sources/mcq/nextvideo/run_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MCQ_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/video_proxy/training/common/gpu_filler_common.sh"

DATA_ROOT="${DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/rollouts}"
NEXTVIDEO_ROOT="${NEXTVIDEO_ROOT:-${DATA_ROOT}/NeXTVideo}"
NEXTVIDEO_INPUT="${NEXTVIDEO_INPUT:-${NEXTVIDEO_ROOT}/train.jsonl}"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROLLOUT_ROOT}/mcq_nextvideo_qwen3_vl_4b_roll8_leq3of8}"

NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-24576}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.82}"
MIN_ACC="${MIN_ACC:-0.0}"
MAX_ACC="${MAX_ACC:-0.375}"
TARGET_TOTAL="${TARGET_TOTAL:-0}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
VERIFY_VIDEOS="${VERIFY_VIDEOS:-true}"
WRITE_KEPT_JSONL="${WRITE_KEPT_JSONL:-false}"

ENABLE_GPU_FILLER="${ENABLE_GPU_FILLER:-false}"
FILLER_LOG_PATH="${FILLER_LOG_PATH:-${OUTPUT_ROOT}/gpu_filler.log}"
FILLER_START_DELAY="${FILLER_START_DELAY:-0}"
FILLER_MODE="${FILLER_MODE:-signal}"
FILLER_PER_GPU="${FILLER_PER_GPU:-true}"
FILLER_SIGNAL_PREFIX="${FILLER_SIGNAL_PREFIX:-/tmp/nextvideo_mcq_gpu_phase_gpu}"
FILLER_TARGET_UTIL="${FILLER_TARGET_UTIL:-80}"
FILLER_BATCH="${FILLER_BATCH:-50}"
FILLER_MATRIX="${FILLER_MATRIX:-8192}"
FILLER_GAP_MATRIX="${FILLER_GAP_MATRIX:-4096}"
FILLER_PUSH_MATRIX="${FILLER_PUSH_MATRIX:-6144}"
FILLER_BUSY_MATRIX="${FILLER_BUSY_MATRIX:-3072}"
FILLER_BUSY_BATCH="${FILLER_BUSY_BATCH:-8}"
FILLER_BUSY_SLEEP_MS="${FILLER_BUSY_SLEEP_MS:-10}"
FILLER_IDLE_SLEEP_MS="${FILLER_IDLE_SLEEP_MS:-6}"
FILLER_ORPHAN_MATRIX="${FILLER_ORPHAN_MATRIX:-4096}"
FILLER_ORPHAN_BATCH="${FILLER_ORPHAN_BATCH:-16}"
FILLER_ORPHAN_SLEEP_MS="${FILLER_ORPHAN_SLEEP_MS:-8}"
FILLER_BUSY_HOLD_MS="${FILLER_BUSY_HOLD_MS:-1600}"
STOP_GPU_FILLER_ON_EXIT="${STOP_GPU_FILLER_ON_EXIT:-false}"

PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_nextvideo.py"
ROLLOUT_SCRIPT="${REPO_ROOT}/video_proxy/training/tools/offline_rollout_filter.py"
REWARD_FN="${MCQ_DIR}/rollout/reward.py:compute_score"
SELECT_SCRIPT="${MCQ_DIR}/rollout/select_from_shards.py"
CONVERT_DIRECT_SCRIPT="${MCQ_DIR}/prepare/convert_to_direct.py"
CHECK_SCRIPT="${MCQ_DIR}/check_format.py"

MCQ_JSONL="${OUTPUT_ROOT}/nextvideo_mcq_all.jsonl"
CONVERT_SUMMARY="${OUTPUT_ROOT}/nextvideo_mcq_all_summary.json"
ROLLOUT_OUTPUT="${OUTPUT_ROOT}/rollout_kept.jsonl"
ROLLOUT_REPORT="${OUTPUT_ROOT}/rollout_report.jsonl"
FINAL_JSONL="${OUTPUT_ROOT}/train_final.jsonl"
FINAL_DIRECT_JSONL="${OUTPUT_ROOT}/train_final_direct.jsonl"
FINAL_CHECK_SUMMARY="${OUTPUT_ROOT}/train_final_direct_check_summary.json"

VISIBLE_GPU_TOKENS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a VISIBLE_GPU_TOKENS <<< "${CUDA_VISIBLE_DEVICES}"
else
    for ((i=0; i<NUM_GPUS; i++)); do
        VISIBLE_GPU_TOKENS+=("${i}")
    done
fi
if (( NUM_GPUS > ${#VISIBLE_GPU_TOKENS[@]} )); then
    echo "NUM_GPUS=${NUM_GPUS}, but only ${#VISIBLE_GPU_TOKENS[@]} visible GPUs are available" >&2
    exit 1
fi

if [[ "${TP_SIZE}" -gt 1 && "${FILLER_PER_GPU}" =~ ^(true|TRUE|1|yes|YES)$ ]]; then
    echo "TP mode detected; disabling per-GPU filler to keep a single shared phase signal."
    FILLER_PER_GPU=false
fi

if [[ -z "${FILLER_GPUS:-}" && "${NUM_GPUS}" -gt 0 ]]; then
    if [[ "${FILLER_PER_GPU}" =~ ^(true|TRUE|1|yes|YES)$ ]]; then
        FILLER_GPUS="$(printf '%s,' "${VISIBLE_GPU_TOKENS[@]:0:${NUM_GPUS}}")"
        FILLER_GPUS="${FILLER_GPUS%,}"
    else
        FILLER_GPUS="$(seq -s, 0 $((NUM_GPUS-1)))"
    fi
fi

echo "============================================="
echo " NeXTVideo MCQ Pipeline"
echo " Dataset root: ${NEXTVIDEO_ROOT}"
echo " Input:        ${NEXTVIDEO_INPUT}"
echo " Model:        ${MODEL_PATH}"
echo " Output:       ${OUTPUT_ROOT}"
echo " GPUs:         ${NUM_GPUS} (TP=${TP_SIZE})"
echo " Rollouts:     ${NUM_ROLLOUTS}"
echo " Acc Range:    [${MIN_ACC}, ${MAX_ACC}]"
echo " Target:       ${TARGET_TOTAL} (0 = keep all filtered)"
echo " Batch:        ${BATCH_SIZE}  Max Tokens: ${MAX_BATCHED_TOKENS}"
echo " Verify videos:${VERIFY_VIDEOS}"
echo " Kept JSONL:   ${WRITE_KEPT_JSONL}"
echo "============================================="

mkdir -p "${OUTPUT_ROOT}"
trap 'gpu_filler_cleanup' EXIT

if [[ "${FORCE}" != "1" && -s "${MCQ_JSONL}" ]]; then
    COUNT=$(wc -l < "${MCQ_JSONL}" | tr -d ' ')
    echo ""
    echo "=== Step 1: prepare_nextvideo [done: ${COUNT} records - skip] ==="
else
    echo ""
    echo "=== Step 1: prepare_nextvideo ==="
    PREPARE_ARGS=(
        --input "${NEXTVIDEO_INPUT}"
        --output "${MCQ_JSONL}"
        --dataset-root "${NEXTVIDEO_ROOT}"
        --split train
        --summary-json "${CONVERT_SUMMARY}"
    )
    if [[ "${VERIFY_VIDEOS,,}" == "true" ]]; then
        PREPARE_ARGS+=(--verify-videos)
    fi
    python3 "${PREPARE_SCRIPT}" "${PREPARE_ARGS[@]}"
fi

ROLLOUT_DONE=0
if [[ "${FORCE}" != "1" && -s "${ROLLOUT_REPORT}" ]]; then
    REPORT_COUNT=$(wc -l < "${ROLLOUT_REPORT}" | tr -d ' ')
    SOURCE_COUNT=$(wc -l < "${MCQ_JSONL}" | tr -d ' ')
    if [[ "${REPORT_COUNT}" -ge "${SOURCE_COUNT}" ]]; then
        echo ""
        echo "=== Step 2: rollout [done: ${REPORT_COUNT}/${SOURCE_COUNT} - skip] ==="
        ROLLOUT_DONE=1
    else
        echo ""
        echo "=== Step 2: rollout report incomplete: ${REPORT_COUNT}/${SOURCE_COUNT}; rerun with FORCE=1 ===" >&2
        exit 1
    fi
fi

if [[ "${ROLLOUT_DONE}" == "0" ]]; then
    SOURCE_COUNT=$(wc -l < "${MCQ_JSONL}" | tr -d ' ')
    echo ""
    echo "=== Step 2: rollout (${SOURCE_COUNT} items × rollout_n=${NUM_ROLLOUTS}) ==="
    gpu_filler_start "[nextvideo-mcq]"

    ROLLOUT_COMMON=(
        --input_jsonl "${MCQ_JSONL}"
        --model_path "${MODEL_PATH}"
        --reward_function "${REWARD_FN}"
        --backend vllm
        --num_rollouts "${NUM_ROLLOUTS}"
        --temperature 0.7
        --max_new_tokens 256
        --gpu_memory_utilization "${GPU_MEM_UTIL}"
        --max_num_batched_tokens "${MAX_BATCHED_TOKENS}"
        --batch_size "${BATCH_SIZE}"
        --min_mean_reward 0.0
        --max_mean_reward 1.0
        --seed "${SEED}"
    )
    ROLLOUT_REPORT_ARGS=(--report_jsonl "${ROLLOUT_REPORT}")
    if [[ "${WRITE_KEPT_JSONL,,}" == "true" ]]; then
        ROLLOUT_OUTPUT_ARGS=(--output_jsonl "${ROLLOUT_OUTPUT}")
    else
        ROLLOUT_OUTPUT_ARGS=()
    fi

    if [[ "${TP_SIZE}" -gt 1 ]]; then
        echo "  TP=${TP_SIZE} mode"
        VERL_GPU_SIGNAL_PATH="${FILLER_SIGNAL_PREFIX}tp" python3 "${ROLLOUT_SCRIPT}" \
            "${ROLLOUT_COMMON[@]}" \
            "${ROLLOUT_REPORT_ARGS[@]}" \
            "${ROLLOUT_OUTPUT_ARGS[@]}" \
            --tensor_parallel_size "${TP_SIZE}"
    elif [[ "${NUM_GPUS}" -gt 1 ]]; then
        echo "  Data-parallel mode (${NUM_GPUS} GPUs)"
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            SHARD_GPU="${VISIBLE_GPU_TOKENS[$i]}"
            SHARD_GPU="${SHARD_GPU//[[:space:]]/}"
            SHARD_SIGNAL_PATH="${FILLER_SIGNAL_PREFIX}${SHARD_GPU}"
            echo "    shard ${i} -> CUDA_VISIBLE_DEVICES=${SHARD_GPU}"
            if [[ "${WRITE_KEPT_JSONL,,}" == "true" ]]; then
                SHARD_OUTPUT_ARGS=(--output_jsonl "${OUTPUT_ROOT}/_shard${i}_kept.jsonl")
            else
                SHARD_OUTPUT_ARGS=()
            fi
            SHARD_REPORT_ARGS=(--report_jsonl "${OUTPUT_ROOT}/_shard${i}_report.jsonl")
            VERL_GPU_SIGNAL_PATH="${SHARD_SIGNAL_PATH}" CUDA_VISIBLE_DEVICES="${SHARD_GPU}" python3 "${ROLLOUT_SCRIPT}" \
                "${ROLLOUT_COMMON[@]}" \
                "${SHARD_REPORT_ARGS[@]}" \
                "${SHARD_OUTPUT_ARGS[@]}" \
                --tensor_parallel_size 1 \
                --shard_id "${i}" --num_shards "${NUM_GPUS}" &
        done
        wait
        cat "${OUTPUT_ROOT}"/_shard*_report.jsonl > "${ROLLOUT_REPORT}"
        if [[ "${WRITE_KEPT_JSONL,,}" == "true" ]]; then
            cat "${OUTPUT_ROOT}"/_shard*_kept.jsonl > "${ROLLOUT_OUTPUT}"
        fi
    else
        echo "  Single GPU mode"
        VERL_GPU_SIGNAL_PATH="${FILLER_SIGNAL_PREFIX}${VISIBLE_GPU_TOKENS[0]}" python3 "${ROLLOUT_SCRIPT}" \
            "${ROLLOUT_COMMON[@]}" \
            "${ROLLOUT_REPORT_ARGS[@]}" \
            "${ROLLOUT_OUTPUT_ARGS[@]}" \
            --tensor_parallel_size 1
    fi
fi

echo ""
echo "=== Step 3: select low-reward records ==="
python3 "${SELECT_SCRIPT}" \
    --input "${MCQ_JSONL}" \
    --report "${ROLLOUT_REPORT}" \
    --output "${FINAL_JSONL}" \
    --summary-json "${OUTPUT_ROOT}/nextvideo_selection_summary.json" \
    --min-mean-reward "${MIN_ACC}" \
    --max-mean-reward "${MAX_ACC}" \
    --target-total "${TARGET_TOTAL}" \
    --seed "${SEED}" \
    --metadata-prefix nextvideo_rollout \
    --training-source nextvideo_rollout_low_reward

echo ""
echo "=== Step 4: convert final to direct-answer format ==="
python3 "${CONVERT_DIRECT_SCRIPT}" "${FINAL_JSONL}" "${FINAL_DIRECT_JSONL}"

echo ""
echo "=== Step 5: check final format ==="
python3 "${CHECK_SCRIPT}" \
    --jsonl "${FINAL_DIRECT_JSONL}" \
    --summary-json "${FINAL_CHECK_SUMMARY}" \
    --min-mean-reward "${MIN_ACC}" \
    --max-mean-reward "${MAX_ACC}"

echo ""
echo "=========================================="
echo " NeXTVideo MCQ pipeline done"
echo " Converted MCQ:  ${MCQ_JSONL}"
echo " Rollout report: ${ROLLOUT_REPORT}"
echo " Final direct:   ${FINAL_DIRECT_JSONL}"
echo " Check summary:  ${FINAL_CHECK_SUMMARY}"
echo "=========================================="
