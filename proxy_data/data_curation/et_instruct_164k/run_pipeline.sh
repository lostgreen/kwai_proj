#!/usr/bin/env bash
# ── ET-Instruct-164K: Duration Filter + Local VLM Screening Pipeline ──
#
# 流程:
#   Step 1: text_filter (仅时长过滤，去掉 event 数限制)
#   Step 2: sample_per_source (每个 source 抽 N 条)
#   Step 3: local_screen Stage 1 (L1/L2 score + domain + quality)
#   Step 4: local_screen Stage 2 (prog_type + visual_diversity + order_dependency)
#
# 用法:
#   bash run_pipeline.sh              # 全量：Step 1-4
#   SKIP_STAGE2=1 bash run_pipeline.sh  # 只跑到 Stage 1
#
# 环境变量 (可选覆盖):
#   ET_JSON_PATH   — et_instruct_164k_txt.json 路径
#   VIDEO_ROOT     — ET-Instruct 视频根目录
#   LOCAL_MODEL    — 本地 VLM 模型路径
#   NUM_GPUS       — local_screen 数据并行 GPU 数 (默认 2)
#   PER_SOURCE     — 每个 source 采样条数 (0 = 全量)
#   OUTPUT_ROOT    — 输出目录
#   SKIP_STAGE2    — 跳过 Stage 2 (1 = 跳过, 默认 0)
#   RESUME         — 断点续跑 (1 = 跳过已有结果, 默认 1)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 配置 ──
ET_JSON_PATH="${ET_JSON_PATH:-/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/et_instruct_164k_txt.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/videos}"
CONFIG="${CONFIG:-../configs/et_instruct_164k.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../results/et_instruct_164k}"
LOCAL_MODEL="${LOCAL_MODEL:-/home/xuboshen/models/Qwen3-VL-4B-Instruct}"
NUM_GPUS="${NUM_GPUS:-2}"
PER_SOURCE="${PER_SOURCE:-0}"
SEED="${SEED:-42}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
RESUME="${RESUME:-1}"

echo "============================================="
echo " ET-Instruct: Duration Filter + Local VLM Screening"
echo " JSON:       $ET_JSON_PATH"
echo " Video Root: $VIDEO_ROOT"
echo " Output:     $OUTPUT_ROOT"
echo " Local VLM:  $LOCAL_MODEL (${NUM_GPUS} GPUs)"
echo " Per Source: $PER_SOURCE (0 = all)"
echo " Skip S2:   $SKIP_STAGE2 (1 = skip)"
echo " Resume:     $RESUME (1 = skip done)"
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
)
if [ "$RESUME" = "1" ]; then
    SCREEN_COMMON+=(--resume)
    # Multi-GPU: each shard reads the merged results file to know what's done
    SCREEN_COMMON+=(--resume_from "$OUTPUT_ROOT/screen_results.jsonl")
fi

# ── Step 3: Stage 1 — L1/L2 + domain + quality ──
echo ""
echo "=== Step 3: local_screen Stage 1 (${NUM_GPUS} GPUs) ==="

if [ "$NUM_GPUS" -gt 1 ]; then
    for i in $(seq 0 $((NUM_GPUS-1))); do
        CUDA_VISIBLE_DEVICES=$i python ../shared/local_screen.py \
            "${SCREEN_COMMON[@]}" \
            --output_jsonl "$OUTPUT_ROOT/screen_shard${i}.jsonl" \
            --keep_jsonl "$OUTPUT_ROOT/keep_shard${i}.jsonl" \
            --reject_jsonl "$OUTPUT_ROOT/reject_shard${i}.jsonl" \
            --shard_id "$i" --num_shards "$NUM_GPUS" &
    done
    wait
    cat "$OUTPUT_ROOT"/keep_shard*.jsonl > "$OUTPUT_ROOT/screen_keep.jsonl"
    cat "$OUTPUT_ROOT"/reject_shard*.jsonl > "$OUTPUT_ROOT/screen_reject.jsonl"
    cat "$OUTPUT_ROOT"/screen_shard*.jsonl > "$OUTPUT_ROOT/screen_results.jsonl"
    # Keep shard files for --resume to work on re-runs
    # (merged file is the source of truth via --resume_from)
else
    python ../shared/local_screen.py \
        "${SCREEN_COMMON[@]}" \
        --output_jsonl "$OUTPUT_ROOT/screen_results.jsonl" \
        --keep_jsonl "$OUTPUT_ROOT/screen_keep.jsonl" \
        --reject_jsonl "$OUTPUT_ROOT/screen_reject.jsonl"
fi

KEEP_S1=$(wc -l < "$OUTPUT_ROOT/screen_keep.jsonl" | tr -d ' ')
REJECT_S1=$(wc -l < "$OUTPUT_ROOT/screen_reject.jsonl" | tr -d ' ')
echo "  Stage 1 → kept=$KEEP_S1  rejected=$REJECT_S1"

# ── Step 4: Stage 2 — prog_type + visual_diversity + order_dependency ──
if [ "$SKIP_STAGE2" != "1" ] && [ "$KEEP_S1" -gt 0 ]; then
    echo ""
    echo "=== Step 4: local_screen Stage 2 (${NUM_GPUS} GPUs) ==="

    if [ "$NUM_GPUS" -gt 1 ]; then
        for i in $(seq 0 $((NUM_GPUS-1))); do
            CUDA_VISIBLE_DEVICES=$i python ../shared/local_screen.py \
                "${SCREEN_COMMON[@]}" \
                --output_jsonl "$OUTPUT_ROOT/s2_shard${i}.jsonl" \
                --keep_jsonl "$OUTPUT_ROOT/s2_keep_shard${i}.jsonl" \
                --reject_jsonl "$OUTPUT_ROOT/s2_reject_shard${i}.jsonl" \
                --s1_keep_jsonl "$OUTPUT_ROOT/screen_keep.jsonl" \
                --s1_reject_jsonl "$OUTPUT_ROOT/screen_reject.jsonl" \
                --shard_id "$i" --num_shards "$NUM_GPUS" \
                --secondary_screen_only &
        done
        wait
        cat "$OUTPUT_ROOT"/s2_keep_shard*.jsonl > "$OUTPUT_ROOT/screen_keep.jsonl"
        # Stage 2 reject = Stage 1 reject + Stage 2 new reject
        # (--secondary_screen_only already merges s1 rejects into reject output)
        cat "$OUTPUT_ROOT"/s2_reject_shard*.jsonl > "$OUTPUT_ROOT/screen_reject.jsonl"
        cat "$OUTPUT_ROOT"/s2_shard*.jsonl > "$OUTPUT_ROOT/screen_results.jsonl"
        # Keep shard files for potential re-runs
    else
        python ../shared/local_screen.py \
            "${SCREEN_COMMON[@]}" \
            --output_jsonl "$OUTPUT_ROOT/screen_results.jsonl" \
            --keep_jsonl "$OUTPUT_ROOT/screen_keep.jsonl" \
            --reject_jsonl "$OUTPUT_ROOT/screen_reject.jsonl" \
            --secondary_screen_only
    fi

    KEEP_S2=$(wc -l < "$OUTPUT_ROOT/screen_keep.jsonl" | tr -d ' ')
    echo "  Stage 2 → kept=$KEEP_S2 (from $KEEP_S1)"
else
    if [ "$SKIP_STAGE2" = "1" ]; then
        echo ""
        echo "  [Skipping Stage 2 (SKIP_STAGE2=1)]"
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
