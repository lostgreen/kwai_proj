#!/usr/bin/env bash
# ── ET-Instruct-164K: Duration Filter + Local VLM Screening Pipeline ──
#
# 新流程:
#   Step 1: text_filter (仅时长过滤，去掉 event 数限制)
#   Step 2: sample_per_source (每个 source 抽 N 条)
#   Step 3: local_screen (本地 Qwen3-VL-4B 预筛选)
#
# 用法:
#   bash run_pipeline.sh
#
# 环境变量 (可选覆盖):
#   ET_JSON_PATH   — et_instruct_164k_txt.json 路径
#   VIDEO_ROOT     — ET-Instruct 视频根目录
#   LOCAL_MODEL    — 本地 VLM 模型路径 (默认 Qwen3-VL-4B-Instruct)
#   NUM_GPUS       — local_screen 数据并行 GPU 数 (默认 1)
#   PER_SOURCE     — 每个 source 采样条数 (0 = 全量，不采样)
#   OUTPUT_ROOT    — 输出目录
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 配置 ──
ET_JSON_PATH="${ET_JSON_PATH:-/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/et_instruct_164k_txt.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/videos}"
CONFIG="${CONFIG:-../configs/et_instruct_164k.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../results/et_instruct_164k}"
LOCAL_MODEL="${LOCAL_MODEL:-/home/xuboshen/models/Qwen3-VL-4B-Instruct}"
NUM_GPUS="${NUM_GPUS:-1}"
PER_SOURCE="${PER_SOURCE:-0}"
SEED="${SEED:-42}"

echo "============================================="
echo " ET-Instruct: Duration Filter + Local VLM Screening"
echo " JSON:       $ET_JSON_PATH"
echo " Video Root: $VIDEO_ROOT"
echo " Output:     $OUTPUT_ROOT"
echo " Local VLM:  $LOCAL_MODEL (${NUM_GPUS} GPUs)"
echo " Per Source: $PER_SOURCE (0 = all)"
echo "============================================="

# ── Step 1: 时长过滤 ──
echo ""
echo "=== Step 1: text_filter (duration-only) ==="
python text_filter.py \
    --json_path "$ET_JSON_PATH" \
    --output_dir "$OUTPUT_ROOT" \
    --config "$CONFIG" \
    --no_event_filter

echo "  → $OUTPUT_ROOT/passed.jsonl"

# ── Step 2: 采样 + 格式转换 ──
echo ""
echo "=== Step 2: sample_per_source ==="
SAMPLE_ARGS=(
    --input "$OUTPUT_ROOT/passed.jsonl"
    --output "$OUTPUT_ROOT/sample_dev.jsonl"
    --video-root "$VIDEO_ROOT"
    --seed "$SEED"
)
if [ "$PER_SOURCE" -gt 0 ]; then
    SAMPLE_ARGS+=(--per-source "$PER_SOURCE")
    echo "  Sampling $PER_SOURCE per source..."
else
    SAMPLE_ARGS+=(--per-source 999999)
    echo "  Using all records (no sampling cap)..."
fi

python sample_per_source.py "${SAMPLE_ARGS[@]}"
echo "  → $OUTPUT_ROOT/sample_dev.jsonl"

# ── Step 3: Local VLM 预筛选 ──
echo ""
echo "=== Step 3: local_screen (${NUM_GPUS} GPUs) ==="
if [ "$NUM_GPUS" -gt 1 ]; then
    for i in $(seq 0 $((NUM_GPUS-1))); do
        CUDA_VISIBLE_DEVICES=$i python local_screen.py \
            --input_jsonl "$OUTPUT_ROOT/sample_dev.jsonl" \
            --output_jsonl "$OUTPUT_ROOT/screen_shard${i}.jsonl" \
            --keep_jsonl "$OUTPUT_ROOT/keep_shard${i}.jsonl" \
            --reject_jsonl "$OUTPUT_ROOT/reject_shard${i}.jsonl" \
            --model_path "$LOCAL_MODEL" \
            --shard_id "$i" --num_shards "$NUM_GPUS" &
    done
    wait
    cat "$OUTPUT_ROOT"/keep_shard*.jsonl > "$OUTPUT_ROOT/screen_keep.jsonl"
    cat "$OUTPUT_ROOT"/reject_shard*.jsonl > "$OUTPUT_ROOT/screen_reject.jsonl"
    cat "$OUTPUT_ROOT"/screen_shard*.jsonl > "$OUTPUT_ROOT/screen_results.jsonl"
    # Clean up shard files
    rm -f "$OUTPUT_ROOT"/keep_shard*.jsonl "$OUTPUT_ROOT"/reject_shard*.jsonl "$OUTPUT_ROOT"/screen_shard*.jsonl
else
    python local_screen.py \
        --input_jsonl "$OUTPUT_ROOT/sample_dev.jsonl" \
        --output_jsonl "$OUTPUT_ROOT/screen_results.jsonl" \
        --keep_jsonl "$OUTPUT_ROOT/screen_keep.jsonl" \
        --reject_jsonl "$OUTPUT_ROOT/screen_reject.jsonl" \
        --model_path "$LOCAL_MODEL"
fi

# ── Summary ──
echo ""
echo "=========================================="
echo " Pipeline 完成!"
KEEP_COUNT=$(wc -l < "$OUTPUT_ROOT/screen_keep.jsonl" | tr -d ' ')
REJECT_COUNT=$(wc -l < "$OUTPUT_ROOT/screen_reject.jsonl" | tr -d ' ')
echo " Kept:     $OUTPUT_ROOT/screen_keep.jsonl ($KEEP_COUNT records)"
echo " Rejected: $OUTPUT_ROOT/screen_reject.jsonl ($REJECT_COUNT records)"
echo " Full:     $OUTPUT_ROOT/screen_results.jsonl"
echo "=========================================="
