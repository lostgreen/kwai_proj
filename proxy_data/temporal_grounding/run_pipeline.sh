#!/bin/bash
# ────────────────────────────────────────────────────────────────
# TG 数据一键流水线: 裁切视频 → 构建数据集 → 验证
#
# 用法:
#   bash run_pipeline.sh [--skip-trim] [--skip-validate]
#
# 环境变量 (可选覆盖):
#   TIMERFT_JSON    train_2k5.json 路径
#   TVGBENCH_JSON   tvgbench.json 路径
#   VIDEO_ROOT      服务器视频根目录
#   OUTPUT_DIR      JSONL 输出目录
#   MAX_DURATION    最大视频时长 (秒)
#   TRIM_WORKERS    ffmpeg 并行数
# ────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 默认参数 ────────────────────────────────────────────────────
TIMERFT_JSON="${TIMERFT_JSON:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset/annotations/train_2k5.json}"
TVGBENCH_JSON="${TVGBENCH_JSON:-}"  # 默认不混入 TVGBench；如需加入: TVGBENCH_JSON=/path/to/tvgbench.json
VIDEO_ROOT="${VIDEO_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/data}"
MAX_DURATION="${MAX_DURATION:-256}"
TRIM_WORKERS="${TRIM_WORKERS:-8}"

# ── 解析参数 ────────────────────────────────────────────────────
SKIP_TRIM=false
SKIP_VALIDATE=false
for arg in "$@"; do
    case "$arg" in
        --skip-trim)     SKIP_TRIM=true ;;
        --skip-validate) SKIP_VALIDATE=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

echo "============================================"
echo "  TG Data Pipeline"
echo "============================================"
echo "  TIMERFT_JSON:  $TIMERFT_JSON"
echo "  TVGBENCH_JSON: ${TVGBENCH_JSON:-<disabled>}"
echo "  VIDEO_ROOT:    $VIDEO_ROOT"
echo "  OUTPUT_DIR:    $OUTPUT_DIR"
echo "  MAX_DURATION:  ${MAX_DURATION}s"
echo "  TRIM_WORKERS:  $TRIM_WORKERS"
echo "============================================"

# ── Step 1: 裁切视频 ────────────────────────────────────────────
if [ "$SKIP_TRIM" = false ]; then
    echo ""
    echo ">>> Step 1: Trimming videos ..."
    TRIM_ARGS="--video-root $VIDEO_ROOT --workers $TRIM_WORKERS"

    # 构建 annotation 参数列表
    ANN_ARGS=""
    [ -f "$TIMERFT_JSON" ]  && ANN_ARGS="$ANN_ARGS $TIMERFT_JSON"
    [ -f "$TVGBENCH_JSON" ] && ANN_ARGS="$ANN_ARGS $TVGBENCH_JSON"

    if [ -z "$ANN_ARGS" ]; then
        echo "[ERROR] No annotation files found!"
        exit 1
    fi

    python "${SCRIPT_DIR}/trim_videos.py" \
        --annotation $ANN_ARGS \
        $TRIM_ARGS

    echo ">>> Step 1 done."
else
    echo ""
    echo ">>> Step 1: SKIPPED (--skip-trim)"
fi

# ── Step 2: 构建数据集 ──────────────────────────────────────────
echo ""
echo ">>> Step 2: Building dataset ..."

BUILD_ARGS="--output_dir $OUTPUT_DIR"
[ -n "$MAX_DURATION" ] && BUILD_ARGS="$BUILD_ARGS --max_duration $MAX_DURATION"
[ -f "$TIMERFT_JSON" ]  && BUILD_ARGS="$BUILD_ARGS --timerft_json $TIMERFT_JSON"
[ -f "$TVGBENCH_JSON" ] && BUILD_ARGS="$BUILD_ARGS --tvgbench_json $TVGBENCH_JSON"

python "${SCRIPT_DIR}/build_dataset.py" \
    --video_base "$VIDEO_ROOT" \
    $BUILD_ARGS

echo ">>> Step 2 done."

# ── Step 3: 验证数据 ────────────────────────────────────────────
if [ "$SKIP_VALIDATE" = false ]; then
    echo ""
    echo ">>> Step 3: Validating videos ..."

    DUR_TAG=""
    [ -n "$MAX_DURATION" ] && DUR_TAG="_max${MAX_DURATION}s"

    TRAIN_JSONL="${OUTPUT_DIR}/tg_train${DUR_TAG}.jsonl"
    VAL_JSONL="${OUTPUT_DIR}/tg_val${DUR_TAG}.jsonl"
    TRAIN_OUT="${OUTPUT_DIR}/tg_train${DUR_TAG}_validated.jsonl"
    VAL_OUT="${OUTPUT_DIR}/tg_val${DUR_TAG}_validated.jsonl"

    python "${SCRIPT_DIR}/validate_tg_videos.py" \
        --input "$TRAIN_JSONL" "$VAL_JSONL" \
        --output "$TRAIN_OUT" "$VAL_OUT" \
        --tolerance 5.0

    # 统计有效数据量
    TRAIN_TOTAL=$(wc -l < "$TRAIN_JSONL")
    TRAIN_VALID=$(wc -l < "$TRAIN_OUT")
    VAL_TOTAL=$(wc -l < "$VAL_JSONL")
    VAL_VALID=$(wc -l < "$VAL_OUT")

    echo ""
    echo "  Validation summary:"
    echo "    Train: $TRAIN_VALID / $TRAIN_TOTAL valid"
    echo "    Val:   $VAL_VALID / $VAL_TOTAL valid"

    echo ">>> Step 3 done."
else
    echo ""
    echo ">>> Step 3: SKIPPED (--skip-validate)"
fi

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
echo "  Output files in: $OUTPUT_DIR"
ls -lh "${OUTPUT_DIR}"/tg_*.jsonl 2>/dev/null || true
