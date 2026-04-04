#!/usr/bin/env bash
# ── TimeLens-100K: Duration Filter + Local VLM Screening Pipeline ──
#
# 流程:
#   Step 1: text_filter (时长 + 事件过滤)
#   Step 2: sample_per_source (格式转换 + 可选采样)
#   Step 3: local_screen Stage 1 (L1/L2 score + domain + quality)
#   Step 4: local_screen Stage 2 (prog_type + visual_diversity + order_dependency)
#
# 自动跳过已完成的步骤：
#   - screen_keep.jsonl 存在 → 跳过 Stage 1
#   - screen_keep.jsonl 含 _screen_2 字段 → 跳过 Stage 2
#
# 用法:
#   bash run_pipeline.sh                # 全量：Step 1-4
#   SKIP_STAGE2=1 bash run_pipeline.sh  # 只跑到 Stage 1
#   FORCE=1 bash run_pipeline.sh        # 强制全量重跑
#
# 环境变量:
#   TL_INPUT       — timelens-100k.jsonl 路径
#   VIDEO_ROOT     — TimeLens 视频根目录
#   LOCAL_MODEL    — 本地 VLM 模型路径
#   NUM_GPUS       — local_screen 数据并行 GPU 数 (默认 2)
#   PER_SOURCE     — 每个 source 采样条数 (0 = 全量)
#   OUTPUT_ROOT    — 输出目录
#   SKIP_STAGE2    — 跳过 Stage 2 (1 = 跳过, 默认 0)
#   FORCE          — 强制重跑所有步骤 (1 = 强制, 默认 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 配置 ──
TL_INPUT="${TL_INPUT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/timelens-100k.jsonl}"
VIDEO_ROOT="${VIDEO_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/video_shards}"
CONFIG="${CONFIG:-../configs/timelens_100k.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../results/timelens_100k}"
LOCAL_MODEL="${LOCAL_MODEL:-/home/xuboshen/models/Qwen3-VL-4B-Instruct}"
LOCAL_SCREEN="../shared/local_screen.py"
NUM_GPUS="${NUM_GPUS:-2}"
PER_SOURCE="${PER_SOURCE:-0}"
SEED="${SEED:-42}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
FORCE="${FORCE:-0}"

echo "============================================="
echo " TimeLens-100K: Duration Filter + Local VLM Screening"
echo " Input:      $TL_INPUT"
echo " Video Root: $VIDEO_ROOT"
echo " Output:     $OUTPUT_ROOT"
echo " Local VLM:  $LOCAL_MODEL (${NUM_GPUS} GPUs)"
echo " Per Source: $PER_SOURCE (0 = all)"
echo " Skip S2:   $SKIP_STAGE2"
echo " Force:     $FORCE"
echo "============================================="

# ── Step 1: 时长 + 事件过滤 ──
echo ""
echo "=== Step 1: text_filter ==="
python text_filter.py \
    --input "$TL_INPUT" \
    --output "$OUTPUT_ROOT/passed_timelens.jsonl" \
    --config "$CONFIG"

echo "  → $OUTPUT_ROOT/passed_timelens.jsonl"

# ── Step 2: 采样 + 格式转换 ──
echo ""
echo "=== Step 2: sample_per_source ==="
SAMPLE_ARGS=(
    --input "$OUTPUT_ROOT/passed_timelens.jsonl"
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

# ── Step 3: Stage 1 ──
_run_stage1() {
    echo ""
    echo "=== Step 3: local_screen Stage 1 (${NUM_GPUS} GPUs) ==="

    if [ "$NUM_GPUS" -gt 1 ]; then
        for i in $(seq 0 $((NUM_GPUS-1))); do
            CUDA_VISIBLE_DEVICES=$i python "$LOCAL_SCREEN" \
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
    else
        python "$LOCAL_SCREEN" \
            "${SCREEN_COMMON[@]}" \
            --output_jsonl "$OUTPUT_ROOT/screen_results.jsonl" \
            --keep_jsonl "$OUTPUT_ROOT/screen_keep.jsonl" \
            --reject_jsonl "$OUTPUT_ROOT/screen_reject.jsonl"
    fi
}

# ── Step 4: Stage 2 ──
_run_stage2() {
    echo ""
    echo "=== Step 4: local_screen Stage 2 (${NUM_GPUS} GPUs) ==="

    if [ "$NUM_GPUS" -gt 1 ]; then
        for i in $(seq 0 $((NUM_GPUS-1))); do
            CUDA_VISIBLE_DEVICES=$i python "$LOCAL_SCREEN" \
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
        cat "$OUTPUT_ROOT"/s2_reject_shard*.jsonl > "$OUTPUT_ROOT/screen_reject.jsonl"
        cat "$OUTPUT_ROOT"/s2_shard*.jsonl > "$OUTPUT_ROOT/screen_results.jsonl"
    else
        python "$LOCAL_SCREEN" \
            "${SCREEN_COMMON[@]}" \
            --output_jsonl "$OUTPUT_ROOT/screen_results.jsonl" \
            --keep_jsonl "$OUTPUT_ROOT/screen_keep.jsonl" \
            --reject_jsonl "$OUTPUT_ROOT/screen_reject.jsonl" \
            --secondary_screen_only
    fi
}

# ── 执行逻辑：检测已有结果，自动跳过 ──

# Stage 1: screen_keep.jsonl 存在且非空 → 已完成
S1_DONE=0
if [ "$FORCE" != "1" ] && [ -s "$OUTPUT_ROOT/screen_keep.jsonl" ]; then
    KEEP_S1=$(wc -l < "$OUTPUT_ROOT/screen_keep.jsonl" | tr -d ' ')
    echo ""
    echo "  [Stage 1 已完成: screen_keep.jsonl 存在, $KEEP_S1 kept — 跳过]"
    S1_DONE=1
else
    _run_stage1
    KEEP_S1=$(wc -l < "$OUTPUT_ROOT/screen_keep.jsonl" | tr -d ' ')
fi
REJECT_S1=$(wc -l < "$OUTPUT_ROOT/screen_reject.jsonl" | tr -d ' ')
echo "  Stage 1 → kept=$KEEP_S1  rejected=$REJECT_S1"

# Stage 2
if [ "$SKIP_STAGE2" != "1" ] && [ "$KEEP_S1" -gt 0 ]; then
    S2_DONE=0
    if [ "$FORCE" != "1" ] && [ "$S1_DONE" = "1" ]; then
        if head -1 "$OUTPUT_ROOT/screen_keep.jsonl" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if '_screen_2' in d else 1)" 2>/dev/null; then
            KEEP_S2=$(wc -l < "$OUTPUT_ROOT/screen_keep.jsonl" | tr -d ' ')
            echo ""
            echo "  [Stage 2 已完成: _screen_2 field present, $KEEP_S2 kept — 跳过]"
            S2_DONE=1
        fi
    fi

    if [ "$S2_DONE" = "0" ]; then
        _run_stage2
        KEEP_S2=$(wc -l < "$OUTPUT_ROOT/screen_keep.jsonl" | tr -d ' ')
    fi
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
