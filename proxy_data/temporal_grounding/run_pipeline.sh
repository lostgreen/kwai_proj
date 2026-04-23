#!/bin/bash
# ────────────────────────────────────────────────────────────────
# TG 数据流水线: 分别对 TimeRFT / TVGBench 做 裁切 → 构建 → 验证
#
# 两个数据源完全独立处理，各自生成带 validated 后缀的 JSONL。
#
# 用法:
#   bash run_pipeline.sh [--skip-trim] [--skip-validate]
#
# 环境变量 (可选覆盖):
#   TIMERFT_JSON    train_2k5.json 路径 (留空则跳过 TimeRFT)
#   TVGBENCH_JSON   tvgbench.json 路径 (留空则跳过 TVGBench)
#   VIDEO_ROOT      服务器视频根目录
#   OUTPUT_DIR      JSONL 输出目录
#   MAX_DURATION    最大视频时长 (秒)
#   TRIM_WORKERS    ffmpeg 并行数
#
# 输出:
#   TimeRFT  → data/tg_timerft_max256s_validated.jsonl
#   TVGBench → data/tg_tvgbench_max256s_validated.jsonl
# ────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# ── 默认参数 ────────────────────────────────────────────────────
TIMERFT_JSON="${TIMERFT_JSON:-${REPO_ROOT}/proxy_data/temporal_grounding/data/train_2k5.json}"
TVGBENCH_JSON="${TVGBENCH_JSON:-${REPO_ROOT}/proxy_data/temporal_grounding/data/tvgbench.json}"
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

DUR_TAG=""
[ -n "$MAX_DURATION" ] && DUR_TAG="_max${MAX_DURATION}s"

echo "============================================"
echo "  TG Data Pipeline (separate sources)"
echo "============================================"
echo "  TIMERFT_JSON:  ${TIMERFT_JSON:-<disabled>}"
echo "  TVGBENCH_JSON: ${TVGBENCH_JSON:-<disabled>}"
echo "  VIDEO_ROOT:    $VIDEO_ROOT"
echo "  OUTPUT_DIR:    $OUTPUT_DIR"
echo "  MAX_DURATION:  ${MAX_DURATION}s"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

# ================================================================
# 通用函数: 处理单个数据源
#   process_source <label> <annotation_json> <source_flag> <output_tag>
#   source_flag: --timerft_json 或 --tvgbench_json
# ================================================================
process_source() {
    local LABEL="$1"
    local ANN_JSON="$2"
    local SOURCE_FLAG="$3"
    local OUT_TAG="$4"

    echo ""
    echo "────────────────────────────────────────"
    echo "  Processing: $LABEL"
    echo "────────────────────────────────────────"

    # Step 1: 裁切视频
    if [ "$SKIP_TRIM" = false ]; then
        echo ">>> [${LABEL}] Step 1: Trimming videos ..."
        python "${SCRIPT_DIR}/trim_videos.py" \
            --annotation "$ANN_JSON" \
            --video-root "$VIDEO_ROOT" \
            --workers "$TRIM_WORKERS"
        echo ">>> [${LABEL}] Step 1 done."
    else
        echo ">>> [${LABEL}] Step 1: SKIPPED (--skip-trim)"
    fi

    # Step 2: 构建数据集 (--n_val 0: 全部输出到 train 文件)
    echo ">>> [${LABEL}] Step 2: Building dataset ..."
    local TMP_DIR="${OUTPUT_DIR}/_tmp_${OUT_TAG}"
    mkdir -p "$TMP_DIR"

    python "${SCRIPT_DIR}/build_dataset.py" \
        "${SOURCE_FLAG}" "$ANN_JSON" \
        --video_base "$VIDEO_ROOT" \
        --output_dir "$TMP_DIR" \
        --max_duration "$MAX_DURATION" \
        --n_val 0

    # build_dataset.py 输出 tg_train{dur_tag}.jsonl (n_val=0 → 全量在 train)
    local RAW_JSONL="${TMP_DIR}/tg_train${DUR_TAG}.jsonl"
    local FINAL_JSONL="${OUTPUT_DIR}/tg_${OUT_TAG}${DUR_TAG}.jsonl"
    mv "$RAW_JSONL" "$FINAL_JSONL"
    rm -rf "$TMP_DIR"
    echo ">>> [${LABEL}] Step 2 done: $(wc -l < "$FINAL_JSONL") records"

    # Step 3: 验证
    if [ "$SKIP_VALIDATE" = false ]; then
        echo ">>> [${LABEL}] Step 3: Validating ..."
        local VALIDATED="${OUTPUT_DIR}/tg_${OUT_TAG}${DUR_TAG}_validated.jsonl"
        python "${SCRIPT_DIR}/validate_tg_videos.py" \
            --input "$FINAL_JSONL" \
            --output "$VALIDATED" \
            --tolerance 1.0

        local TOTAL
        TOTAL=$(wc -l < "$FINAL_JSONL")
        local VALID
        VALID=$(wc -l < "$VALIDATED")
        echo "  [${LABEL}] Validated: $VALID / $TOTAL"
        echo ">>> [${LABEL}] Step 3 done."
    else
        echo ">>> [${LABEL}] Step 3: SKIPPED (--skip-validate)"
    fi
}

# ================================================================
# 分别处理 TimeRFT 和 TVGBench
# ================================================================

if [ -n "$TIMERFT_JSON" ] && [ -f "$TIMERFT_JSON" ]; then
    process_source "TimeRFT" "$TIMERFT_JSON" "--timerft_json" "timerft"
else
    echo ""
    echo ">>> TimeRFT: SKIPPED (TIMERFT_JSON not set or file not found)"
fi

if [ -n "$TVGBENCH_JSON" ] && [ -f "$TVGBENCH_JSON" ]; then
    process_source "TVGBench" "$TVGBENCH_JSON" "--tvgbench_json" "tvgbench"
else
    echo ""
    echo ">>> TVGBench: SKIPPED (TVGBENCH_JSON not set or file not found)"
fi

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "============================================"
echo "  Output files in: $OUTPUT_DIR"
ls -lh "${OUTPUT_DIR}"/tg_*.jsonl 2>/dev/null || true
