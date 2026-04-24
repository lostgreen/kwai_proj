#!/usr/bin/env bash
# ── TimeLens Short-Video TG Rollout Pipeline ──
#
# Flow:
#   Step 1: build_tg_rollout_dataset.py — expand raw TimeLens records to query-level TG JSONL
#   Step 2: offline_rollout_filter.py   — Qwen3-VL sampled rollout (report keeps all items)
#
# Notes:
#   - rollout_report.jsonl is the main artifact for later mean-IoU analysis
#   - current offline_rollout_filter does NOT pass duration metadata into TG reward,
#     so report.mean_reward is effectively raw temporal IoU for this workflow
#
# Usage (from train/):
#   bash proxy_data/data_curation/timelens_100k/run_tg_rollout.sh
#
# Key env vars:
#   INPUT_RAW       — raw TimeLens JSONL (default short_pool_raw.jsonl)
#   VIDEO_ROOT      — TimeLens video root
#   MODEL_PATH      — Qwen3-VL model path
#   OUTPUT_ROOT     — output directory
#   NUM_GPUS        — data-parallel GPU count (default 8)
#   TP_SIZE         — tensor parallel size (default 1; TP>1 uses a single engine)
#   NUM_ROLLOUTS    — sampled generations per query (default 8)
#   BATCH_SIZE      — vLLM mini-batch size
#   ENABLE_GPU_FILLER / FILLER_* — keep GPU avg util high during rollout
#   RUN_ANALYSIS    — generate score/duration analysis after rollout (default true)
#   FORCE           — force rebuild/re-rollout (1 = rerun)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

source "${REPO_ROOT}/local_scripts/gpu_filler_common.sh"

BUILD_SCRIPT="${SCRIPT_DIR}/build_tg_rollout_dataset.py"
ANALYSIS_SCRIPT="${SCRIPT_DIR}/analyze_tg_rollout.py"
ROLLOUT_SCRIPT="${REPO_ROOT}/local_scripts/offline_rollout_filter.py"
RESUME_HELPER="${REPO_ROOT}/proxy_data/llava_video_178k/resume_helper.py"
REWARD_FN="${REPO_ROOT}/verl/reward_function/temporal_grounding_reward.py:compute_score"

INPUT_RAW="${INPUT_RAW:-${REPO_ROOT}/proxy_data/data_curation/results/timelens_100k_short/short_pool_raw.jsonl}"
VIDEO_ROOT="${VIDEO_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/video_shards}"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/proxy_data/data_curation/results/timelens_100k_short/tg_rollout_qwen3_vl_8b_roll8}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$OUTPUT_ROOT/analysis}"

NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-24576}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.82}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
MAX_QUERIES="${MAX_QUERIES:-0}"
FORCE="${FORCE:-0}"
RUN_ANALYSIS="${RUN_ANALYSIS:-true}"

ENABLE_GPU_FILLER="${ENABLE_GPU_FILLER:-true}"
FILLER_LOG_PATH="${FILLER_LOG_PATH:-$OUTPUT_ROOT/gpu_filler.log}"
FILLER_START_DELAY="${FILLER_START_DELAY:-0}"
FILLER_MODE="${FILLER_MODE:-signal}"
FILLER_PER_GPU="${FILLER_PER_GPU:-true}"
FILLER_SIGNAL_PREFIX="${FILLER_SIGNAL_PREFIX:-/tmp/timelens_tg_gpu_phase_gpu}"
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

QUERY_JSONL="$OUTPUT_ROOT/tg_rollout_input.jsonl"
ROLLOUT_OUTPUT="$OUTPUT_ROOT/rollout_kept.jsonl"
ROLLOUT_REPORT="$OUTPUT_ROOT/rollout_report.jsonl"

echo "============================================="
echo " TimeLens TG Rollout"
echo " Input raw:   $INPUT_RAW"
echo " Video root:  $VIDEO_ROOT"
echo " Model:       $MODEL_PATH"
echo " Output:      $OUTPUT_ROOT"
if [ "$TP_SIZE" -gt 1 ]; then
    echo " GPUs:        TP=$TP_SIZE"
else
    echo " GPUs:        DP=${NUM_GPUS}"
fi
echo " Rollouts:    $NUM_ROLLOUTS"
echo " Batch:       $BATCH_SIZE  Max Tokens: $MAX_BATCHED_TOKENS"
echo " Filler:      enabled=$ENABLE_GPU_FILLER mode=$FILLER_MODE per_gpu=$FILLER_PER_GPU target=$FILLER_TARGET_UTIL"
echo "============================================="

mkdir -p "$OUTPUT_ROOT"
trap 'gpu_filler_cleanup' EXIT

if [ ! -f "$INPUT_RAW" ]; then
    echo "Input raw JSONL not found: $INPUT_RAW" >&2
    exit 1
fi

if [ "$FORCE" != "1" ] && [ -s "$QUERY_JSONL" ]; then
    COUNT=$(wc -l < "$QUERY_JSONL" | tr -d ' ')
    echo ""
    echo "=== Step 1: build_tg_rollout_dataset [已完成: $COUNT query items — 跳过] ==="
else
    echo ""
    echo "=== Step 1: build_tg_rollout_dataset ==="
    BUILD_ARGS=(
        --input "$INPUT_RAW"
        --output "$QUERY_JSONL"
        --video-root "$VIDEO_ROOT"
    )
    if [ "$MAX_VIDEOS" -gt 0 ]; then
        BUILD_ARGS+=(--max-videos "$MAX_VIDEOS")
    fi
    if [ "$MAX_QUERIES" -gt 0 ]; then
        BUILD_ARGS+=(--max-queries "$MAX_QUERIES")
    fi
    python "$BUILD_SCRIPT" "${BUILD_ARGS[@]}"
fi

ROLLOUT_DONE=0
RESUME_INPUT=""
SOURCE_COUNT=$(wc -l < "$QUERY_JSONL" | tr -d ' ')

if [ "$FORCE" != "1" ] && [ -s "$ROLLOUT_REPORT" ]; then
    REPORT_COUNT=$(wc -l < "$ROLLOUT_REPORT" | tr -d ' ')
    if [ "$REPORT_COUNT" -ge "$SOURCE_COUNT" ]; then
        echo ""
        echo "=== Step 2: rollout [已完成: $REPORT_COUNT/$SOURCE_COUNT — 跳过] ==="
        ROLLOUT_DONE=1
    else
        echo ""
        echo "=== Step 2: rollout [断点续传: 已完成 $REPORT_COUNT/$SOURCE_COUNT] ==="
        RESUME_INPUT="$OUTPUT_ROOT/_remaining.jsonl"
        python "$RESUME_HELPER" \
            --input "$QUERY_JSONL" \
            --report "$ROLLOUT_REPORT" \
            --output "$RESUME_INPUT" || {
                RESUME_EXIT=$?
                if [ "$RESUME_EXIT" = "42" ]; then
                    echo "  All items already processed!"
                    ROLLOUT_DONE=1
                else
                    echo "  Resume helper failed (exit=$RESUME_EXIT)" >&2
                    exit 1
                fi
            }
    fi
fi

if [ "$ROLLOUT_DONE" = "0" ]; then
    EFFECTIVE_INPUT="${RESUME_INPUT:-$QUERY_JSONL}"
    REMAINING_COUNT=$(wc -l < "$EFFECTIVE_INPUT" | tr -d ' ')
    echo ""
    echo "=== Step 2: rollout (${REMAINING_COUNT} query items × rollout_n=$NUM_ROLLOUTS) ==="

    gpu_filler_start "[timelens-tg]"

    if [ -n "$RESUME_INPUT" ]; then
        RUN_OUTPUT="$OUTPUT_ROOT/_resume_kept.jsonl"
        RUN_REPORT="$OUTPUT_ROOT/_resume_report.jsonl"
    else
        RUN_OUTPUT="$ROLLOUT_OUTPUT"
        RUN_REPORT="$ROLLOUT_REPORT"
    fi

    ROLLOUT_COMMON=(
        --input_jsonl "$EFFECTIVE_INPUT"
        --output_jsonl "$RUN_OUTPUT"
        --report_jsonl "$RUN_REPORT"
        --model_path "$MODEL_PATH"
        --reward_function "$REWARD_FN"
        --backend vllm
        --num_rollouts "$NUM_ROLLOUTS"
        --temperature "$TEMPERATURE"
        --top_p "$TOP_P"
        --max_new_tokens "$MAX_NEW_TOKENS"
        --gpu_memory_utilization "$GPU_MEM_UTIL"
        --max_num_batched_tokens "$MAX_BATCHED_TOKENS"
        --batch_size "$BATCH_SIZE"
        --min_mean_reward 0.0
        --max_mean_reward 1.0
        --seed "$SEED"
    )

    if [ "$MAX_SAMPLES" -gt 0 ]; then
        ROLLOUT_COMMON+=(--max_samples "$MAX_SAMPLES")
    fi

    if [ "$TP_SIZE" -gt 1 ]; then
        echo "  TP=$TP_SIZE mode"
        VERL_GPU_SIGNAL_PATH="${FILLER_SIGNAL_PREFIX}tp" python "$ROLLOUT_SCRIPT" \
            "${ROLLOUT_COMMON[@]}" \
            --tensor_parallel_size "$TP_SIZE"
    elif [ "$NUM_GPUS" -gt 1 ]; then
        echo "  Data-parallel mode (${NUM_GPUS} GPUs)"
        for i in $(seq 0 $((NUM_GPUS-1))); do
            SHARD_GPU="${VISIBLE_GPU_TOKENS[$i]}"
            SHARD_GPU="${SHARD_GPU//[[:space:]]/}"
            SHARD_SIGNAL_PATH="${FILLER_SIGNAL_PREFIX}${SHARD_GPU}"
            echo "    shard ${i} -> CUDA_VISIBLE_DEVICES=${SHARD_GPU}"
            VERL_GPU_SIGNAL_PATH="${SHARD_SIGNAL_PATH}" CUDA_VISIBLE_DEVICES="${SHARD_GPU}" python "$ROLLOUT_SCRIPT" \
                "${ROLLOUT_COMMON[@]}" \
                --tensor_parallel_size 1 \
                --shard_id "$i" --num_shards "$NUM_GPUS" \
                --output_jsonl "$OUTPUT_ROOT/_shard${i}_kept.jsonl" \
                --report_jsonl "$OUTPUT_ROOT/_shard${i}_report.jsonl" &
        done
        wait
        cat "$OUTPUT_ROOT"/_shard*_kept.jsonl > "$RUN_OUTPUT"
        cat "$OUTPUT_ROOT"/_shard*_report.jsonl > "$RUN_REPORT"
        rm -f "$OUTPUT_ROOT"/_shard*_kept.jsonl "$OUTPUT_ROOT"/_shard*_report.jsonl
    else
        echo "  Single GPU mode"
        VERL_GPU_SIGNAL_PATH="${FILLER_SIGNAL_PREFIX}${VISIBLE_GPU_TOKENS[0]}" python "$ROLLOUT_SCRIPT" \
            "${ROLLOUT_COMMON[@]}" \
            --tensor_parallel_size 1
    fi

    if [ -n "$RESUME_INPUT" ] && [ -s "$RUN_REPORT" ]; then
        echo "  Merging resume results..."
        cat "$RUN_OUTPUT" >> "$ROLLOUT_OUTPUT"
        cat "$RUN_REPORT" >> "$ROLLOUT_REPORT"
        rm -f "$RESUME_INPUT" "$RUN_OUTPUT" "$RUN_REPORT"
        MERGED_COUNT=$(wc -l < "$ROLLOUT_REPORT" | tr -d ' ')
        echo "  Merged report: $MERGED_COUNT total records"
    fi
fi

case "${RUN_ANALYSIS}" in
    true|TRUE|1|yes|YES)
        ANALYSIS_REPORT_ARGS=()
        if [ -s "$ROLLOUT_REPORT" ]; then
            ANALYSIS_REPORT_ARGS=(--report "$ROLLOUT_REPORT")
        else
            shopt -s nullglob
            SHARD_REPORTS=("$OUTPUT_ROOT"/_shard*_report.jsonl)
            shopt -u nullglob
            if [ "${#SHARD_REPORTS[@]}" -gt 0 ]; then
                ANALYSIS_REPORT_ARGS=(--report "${SHARD_REPORTS[@]}")
                echo "  Final rollout_report.jsonl not found; using ${#SHARD_REPORTS[@]} shard reports for analysis."
            fi
        fi

        if [ "${#ANALYSIS_REPORT_ARGS[@]}" -gt 0 ] && [ -s "$QUERY_JSONL" ]; then
            echo ""
            echo "=== Step 3: analyze rollout score/duration distribution ==="
            python "$ANALYSIS_SCRIPT" \
                "${ANALYSIS_REPORT_ARGS[@]}" \
                --input-jsonl "$QUERY_JSONL" \
                --output-dir "$ANALYSIS_DIR"
        else
            echo ""
            echo "=== Step 3: analyze rollout [skip: missing report or query input] ==="
        fi
        ;;
    *)
        echo ""
        echo "=== Step 3: analyze rollout [disabled] ==="
        ;;
esac

echo ""
echo "=========================================="
echo " Rollout done!"
if [ -s "$QUERY_JSONL" ]; then
    QUERY_COUNT=$(wc -l < "$QUERY_JSONL" | tr -d ' ')
    echo " Query input: $QUERY_JSONL ($QUERY_COUNT items)"
fi
if [ -s "$ROLLOUT_REPORT" ]; then
    REPORT_COUNT=$(wc -l < "$ROLLOUT_REPORT" | tr -d ' ')
    echo " Report:      $ROLLOUT_REPORT ($REPORT_COUNT items)"
fi
if [ -s "$ROLLOUT_OUTPUT" ]; then
    KEEP_COUNT=$(wc -l < "$ROLLOUT_OUTPUT" | tr -d ' ')
    echo " Diverse:     $ROLLOUT_OUTPUT ($KEEP_COUNT items)"
fi
if [ -d "$ANALYSIS_DIR" ]; then
    echo " Analysis:    $ANALYSIS_DIR"
fi
echo "=========================================="
