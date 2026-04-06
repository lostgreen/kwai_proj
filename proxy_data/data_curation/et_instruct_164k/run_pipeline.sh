#!/usr/bin/env bash
# ── ET-Instruct-164K: Duration Filter + Local VLM Screening Pipeline ──
#
# 流程:
#   Step 1: text_filter (仅时长过滤)
#   Step 2: sample_per_source (格式转换 + 可选采样)
#   Step 3: local_screen (unified single-pass: L1/L2 + prog_type + order + domain + quality)
#
# 用法 (从 train/ 目录运行):
#   bash proxy_data/data_curation/et_instruct_164k/run_pipeline.sh
#   LOCAL_MODEL=/m2v_intern/xuboshen/zgw/models/Qwen3-VL-32B-Instruct TP_SIZE=2 \
#     bash proxy_data/data_curation/et_instruct_164k/run_pipeline.sh
#   FORCE=1 bash proxy_data/data_curation/et_instruct_164k/run_pipeline.sh
#
# 环境变量:
#   ET_JSON_PATH   — et_instruct_164k_txt.json 路径
#   VIDEO_ROOT     — ET-Instruct 视频根目录
#   LOCAL_MODEL    — 本地 VLM 模型路径
#   TP_SIZE        — tensor parallel size (默认 1; >1 时用 TP 模式而非数据并行)
#   NUM_GPUS       — 数据并行 GPU 数 (仅 TP_SIZE=1 时生效, 默认 2)
#   GPU_MEM_UTIL   — gpu_memory_utilization (默认 0.85)
#   BATCH_SIZE     — inference batch size (默认 8)
#   MAX_BATCHED_TOKENS — vLLM max_num_batched_tokens (默认 16384)
#   PER_SOURCE     — 每个 source 采样条数 (0 = 全量)
#   OUTPUT_ROOT    — 输出目录
#   FORCE          — 强制重跑 (1 = 强制, 默认 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 配置 ──
ET_JSON_PATH="${ET_JSON_PATH:-/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/et_instruct_164k_txt.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/videos}"
CONFIG="${CONFIG:-../configs/et_instruct_164k.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../results/et_instruct_164k}"
LOCAL_MODEL="${LOCAL_MODEL:-/home/xuboshen/models/Qwen3-VL-4B-Instruct}"
LOCAL_SCREEN="../shared/local_screen.py"
TP_SIZE="${TP_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
PER_SOURCE="${PER_SOURCE:-0}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"

echo "============================================="
echo " ET-Instruct: Duration Filter + Local VLM Screening"
echo " JSON:       $ET_JSON_PATH"
echo " Video Root: $VIDEO_ROOT"
echo " Output:     $OUTPUT_ROOT"
if [ "$TP_SIZE" -gt 1 ]; then
    echo " Model:      $LOCAL_MODEL (TP=$TP_SIZE)"
else
    echo " Model:      $LOCAL_MODEL (DP=${NUM_GPUS} GPUs)"
fi
echo " Batch:      $BATCH_SIZE  Max Tokens: $MAX_BATCHED_TOKENS"
echo " Per Source: $PER_SOURCE (0 = all)"
echo " Force:     $FORCE"
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

# ── 共用参数 ──
SCREEN_COMMON=(
    --input_jsonl "$OUTPUT_ROOT/sample_dev.jsonl"
    --model_path "$LOCAL_MODEL"
    --gpu_memory_utilization "$GPU_MEM_UTIL"
    --batch_size "$BATCH_SIZE"
    --max_num_batched_tokens "$MAX_BATCHED_TOKENS"
    --unified
)
if [ "$TP_SIZE" -gt 1 ]; then
    SCREEN_COMMON+=(--tensor_parallel_size "$TP_SIZE")
fi

# ── Step 3: Unified screening ──
SCREEN_DONE=0
if [ "$FORCE" != "1" ] && [ -s "$OUTPUT_ROOT/screen_keep.jsonl" ]; then
    if head -1 "$OUTPUT_ROOT/screen_keep.jsonl" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('_screen',{}).get('prog_type') else 1)" 2>/dev/null; then
        KEEP=$(wc -l < "$OUTPUT_ROOT/screen_keep.jsonl" | tr -d ' ')
        echo ""
        echo "  [Screening 已完成: $KEEP kept — 跳过. Use FORCE=1 to rerun]"
        SCREEN_DONE=1
    fi
fi

if [ "$SCREEN_DONE" = "0" ]; then
    echo ""
    echo "=== Step 3: local_screen unified ==="

    if [ "$TP_SIZE" -gt 1 ]; then
        echo "  TP=$TP_SIZE mode"
        python "$LOCAL_SCREEN" \
            "${SCREEN_COMMON[@]}" \
            --output_jsonl "$OUTPUT_ROOT/screen_results.jsonl" \
            --keep_jsonl "$OUTPUT_ROOT/screen_keep.jsonl" \
            --reject_jsonl "$OUTPUT_ROOT/screen_reject.jsonl"
    elif [ "$NUM_GPUS" -gt 1 ]; then
        echo "  Data-parallel mode (${NUM_GPUS} GPUs)"
        for i in $(seq 0 $((NUM_GPUS-1))); do
            CUDA_VISIBLE_DEVICES=$i python "$LOCAL_SCREEN" \
                "${SCREEN_COMMON[@]}" \
                --shard_id "$i" --num_shards "$NUM_GPUS" \
                --output_jsonl "$OUTPUT_ROOT/screen_shard${i}.jsonl" \
                --keep_jsonl "$OUTPUT_ROOT/keep_shard${i}.jsonl" \
                --reject_jsonl "$OUTPUT_ROOT/reject_shard${i}.jsonl" &
        done
        wait
        cat "$OUTPUT_ROOT"/keep_shard*.jsonl > "$OUTPUT_ROOT/screen_keep.jsonl"
        cat "$OUTPUT_ROOT"/reject_shard*.jsonl > "$OUTPUT_ROOT/screen_reject.jsonl"
        cat "$OUTPUT_ROOT"/screen_shard*.jsonl > "$OUTPUT_ROOT/screen_results.jsonl"
    else
        echo "  Single GPU mode"
        python "$LOCAL_SCREEN" \
            "${SCREEN_COMMON[@]}" \
            --output_jsonl "$OUTPUT_ROOT/screen_results.jsonl" \
            --keep_jsonl "$OUTPUT_ROOT/screen_keep.jsonl" \
            --reject_jsonl "$OUTPUT_ROOT/screen_reject.jsonl"
    fi
fi

# ── Summary ──
echo ""
echo "=========================================="
echo " Pipeline done!"
KEEP_COUNT=$(wc -l < "$OUTPUT_ROOT/screen_keep.jsonl" | tr -d ' ')
REJECT_COUNT=$(wc -l < "$OUTPUT_ROOT/screen_reject.jsonl" | tr -d ' ')
echo " Kept:     $OUTPUT_ROOT/screen_keep.jsonl ($KEEP_COUNT records)"
echo " Rejected: $OUTPUT_ROOT/screen_reject.jsonl ($REJECT_COUNT records)"
echo " Full:     $OUTPUT_ROOT/screen_results.jsonl"
echo "=========================================="
