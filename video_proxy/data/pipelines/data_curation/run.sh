#!/usr/bin/env bash
# Duration-first data curation.
#
# Examples:
#   DATASET=et_instruct_164k INPUT=/path/to/et_instruct_164k_txt.json \
#     VIDEO_ROOT=/path/to/videos bash video_proxy/data/pipelines/data_curation/run.sh
#
#   DATASET=timelens_100k INPUT=/path/to/timelens-100k.jsonl \
#     VIDEO_ROOT=/path/to/video_shards TARGET_TOTAL=3000 BALANCED_TOTAL=1 \
#     bash video_proxy/data/pipelines/data_curation/run.sh
#
# Optional local scoring:
#   LOCAL_SCORE=1 LOCAL_MODEL=/path/to/Qwen3-VL-4B-Instruct NUM_GPUS=2 bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-et_instruct_164k}"
MIN_DURATION="${MIN_DURATION:-60}"
MAX_DURATION="${MAX_DURATION:-240}"
PER_SOURCE="${PER_SOURCE:-0}"
TARGET_TOTAL="${TARGET_TOTAL:-0}"
BALANCED_TOTAL="${BALANCED_TOTAL:-0}"
SEED="${SEED:-42}"
LOCAL_SCORE="${LOCAL_SCORE:-0}"

case "$DATASET" in
    et_instruct_164k)
        INPUT="${INPUT:-/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/et_instruct_164k_txt.json}"
        VIDEO_ROOT="${VIDEO_ROOT:-/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/videos}"
        OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/results/et_instruct_164k}"
        ;;
    timelens_100k)
        INPUT="${INPUT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/timelens-100k.jsonl}"
        VIDEO_ROOT="${VIDEO_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/video_shards}"
        OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/results/timelens_100k}"
        ;;
    *)
        echo "Unsupported DATASET=$DATASET (expected et_instruct_164k or timelens_100k)" >&2
        exit 2
        ;;
esac

echo "============================================="
echo " Data curation: duration-first"
echo " Dataset:      $DATASET"
echo " Input:        $INPUT"
echo " Video Root:   $VIDEO_ROOT"
echo " Output:       $OUTPUT_ROOT"
echo " Duration:     ${MIN_DURATION}s - ${MAX_DURATION}s"
echo " Per Source:   $PER_SOURCE"
echo " Target:       $TARGET_TOTAL (balanced=$BALANCED_TOTAL)"
echo " Local Score:  $LOCAL_SCORE"
echo "============================================="

FILTER_ARGS=(
    --dataset "$DATASET"
    --input "$INPUT"
    --output-dir "$OUTPUT_ROOT"
    --video-root "$VIDEO_ROOT"
    --min-duration "$MIN_DURATION"
    --max-duration "$MAX_DURATION"
    --per-source "$PER_SOURCE"
    --target-total "$TARGET_TOTAL"
    --seed "$SEED"
)
if [ "$BALANCED_TOTAL" = "1" ]; then
    FILTER_ARGS+=(--balanced-total)
fi

python -m video_proxy.data.pipelines.data_curation.curation.duration_filter "${FILTER_ARGS[@]}"

if [ "$LOCAL_SCORE" = "1" ]; then
    LOCAL_MODEL="${LOCAL_MODEL:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
    TP_SIZE="${TP_SIZE:-1}"
    NUM_GPUS="${NUM_GPUS:-1}"
    GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"

    if [ "$TP_SIZE" -gt 1 ]; then
        python -m video_proxy.data.pipelines.data_curation.curation.local_score \
            --input-jsonl "$OUTPUT_ROOT/duration_keep.jsonl" \
            --output-jsonl "$OUTPUT_ROOT/screen_results.jsonl" \
            --keep-jsonl "$OUTPUT_ROOT/screen_keep.jsonl" \
            --reject-jsonl "$OUTPUT_ROOT/screen_reject.jsonl" \
            --model-path "$LOCAL_MODEL" \
            --tensor-parallel-size "$TP_SIZE" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --batch-size "$BATCH_SIZE" \
            --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    elif [ "$NUM_GPUS" -gt 1 ]; then
        for i in $(seq 0 $((NUM_GPUS-1))); do
            CUDA_VISIBLE_DEVICES=$i python -m video_proxy.data.pipelines.data_curation.curation.local_score \
                --input-jsonl "$OUTPUT_ROOT/duration_keep.jsonl" \
                --output-jsonl "$OUTPUT_ROOT/screen_shard${i}.jsonl" \
                --keep-jsonl "$OUTPUT_ROOT/keep_shard${i}.jsonl" \
                --reject-jsonl "$OUTPUT_ROOT/reject_shard${i}.jsonl" \
                --model-path "$LOCAL_MODEL" \
                --shard-id "$i" \
                --num-shards "$NUM_GPUS" \
                --gpu-memory-utilization "$GPU_MEM_UTIL" \
                --batch-size "$BATCH_SIZE" \
                --max-num-batched-tokens "$MAX_BATCHED_TOKENS" &
        done
        wait
        cat "$OUTPUT_ROOT"/screen_shard*.jsonl > "$OUTPUT_ROOT/screen_results.jsonl"
        cat "$OUTPUT_ROOT"/keep_shard*.jsonl > "$OUTPUT_ROOT/screen_keep.jsonl"
        cat "$OUTPUT_ROOT"/reject_shard*.jsonl > "$OUTPUT_ROOT/screen_reject.jsonl"
    else
        python -m video_proxy.data.pipelines.data_curation.curation.local_score \
            --input-jsonl "$OUTPUT_ROOT/duration_keep.jsonl" \
            --output-jsonl "$OUTPUT_ROOT/screen_results.jsonl" \
            --keep-jsonl "$OUTPUT_ROOT/screen_keep.jsonl" \
            --reject-jsonl "$OUTPUT_ROOT/screen_reject.jsonl" \
            --model-path "$LOCAL_MODEL" \
            --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --batch-size "$BATCH_SIZE" \
            --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
    fi
fi

echo "============================================="
echo " Data curation done"
echo " Duration keep: $OUTPUT_ROOT/duration_keep.jsonl"
echo " Screen keep:   $OUTPUT_ROOT/screen_keep.jsonl"
echo " Summary:       $OUTPUT_ROOT/duration_summary.json"
echo "============================================="
