#!/usr/bin/env bash
# ── ET-Instruct 重新筛选 + 采样 + 标注试跑 Pipeline ──
#
# 目标: 只按时长过滤 (去掉 event 数限制)，每个 source 抽 2 条试标注
#
# 用法:
#   bash run_sample_annotate.sh
#
# 环境变量 (可选覆盖):
#   ET_JSON_PATH   — et_instruct_164k_txt.json 路径
#   VIDEO_ROOT     — ET-Instruct 视频根目录
#   API_BASE       — VLM API base URL
#   MODEL          — VLM 模型名
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 配置 ──
ET_JSON_PATH="${ET_JSON_PATH:-/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/et_instruct_164k_txt.json}"
VIDEO_ROOT="${VIDEO_ROOT:-/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/videos}"
CONFIG="${CONFIG:-../configs/et_instruct_164k.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../results/et_instruct_164k_duration_only}"

API_BASE="${API_BASE:-https://api.novita.ai/v3/openai}"
MODEL="${MODEL:-pa/gmn-2.5-pr}"
WORKERS="${WORKERS:-4}"
PER_SOURCE="${PER_SOURCE:-2}"

# 标注脚本路径
HIER_SEG_DIR="../../youcook2_seg/hier_seg_annotation"

echo "============================================="
echo " ET-Instruct: Duration-Only Filter + Sample + Annotate"
echo " JSON:       $ET_JSON_PATH"
echo " Video Root: $VIDEO_ROOT"
echo " Output:     $OUTPUT_ROOT"
echo " Per Source: $PER_SOURCE"
echo " API:        $API_BASE / $MODEL"
echo "============================================="

# ── Step 1: 文本筛选 (只过滤时长) ──
echo ""
echo "=== Step 1: text_filter (duration-only, --no_event_filter) ==="
python text_filter.py \
    --json_path "$ET_JSON_PATH" \
    --output_dir "$OUTPUT_ROOT" \
    --config "$CONFIG" \
    --no_event_filter

echo ""
echo "passed.jsonl → $OUTPUT_ROOT/passed.jsonl"

# ── Step 2: 每个 source 采样 ──
echo ""
echo "=== Step 2: sample_per_source (${PER_SOURCE} per source) ==="
python sample_per_source.py \
    --input "$OUTPUT_ROOT/passed.jsonl" \
    --output "$OUTPUT_ROOT/sample_dev.jsonl" \
    --per-source "$PER_SOURCE" \
    --video-root "$VIDEO_ROOT" \
    --seed 42

echo ""
echo "sample_dev.jsonl → $OUTPUT_ROOT/sample_dev.jsonl"

# ── Step 3: 抽帧 (1fps, 全视频) ──
FRAMES_DIR="$OUTPUT_ROOT/frames"
echo ""
echo "=== Step 3: extract_frames (1fps full video → $FRAMES_DIR) ==="
python "$HIER_SEG_DIR/extract_frames.py" \
    --jsonl "$OUTPUT_ROOT/sample_dev.jsonl" \
    --output-dir "$FRAMES_DIR" \
    --fps 1 \
    --min-frames 16 \
    --workers "$WORKERS"

# ── Step 4: Merged L1+L2+Topology 标注 ──
ANN_DIR="$OUTPUT_ROOT/annotations"
echo ""
echo "=== Step 4: annotate --level merged ==="
python "$HIER_SEG_DIR/annotate.py" \
    --frames-dir "$FRAMES_DIR" \
    --output-dir "$ANN_DIR" \
    --level merged \
    --api-base "$API_BASE" \
    --model "$MODEL" \
    --workers "$WORKERS"

# ── Step 5: L3 抽帧 (2fps, per-event/phase) ──
FRAMES_L3_DIR="$OUTPUT_ROOT/frames_l3"
echo ""
echo "=== Step 5: extract L3 frames (2fps per-event) ==="
python "$HIER_SEG_DIR/extract_frames.py" \
    --annotation-dir "$ANN_DIR" \
    --original-video-root "$VIDEO_ROOT" \
    --output-dir "$FRAMES_L3_DIR" \
    --fps 2 \
    --workers "$WORKERS"

# ── Step 6: L3 标注 ──
echo ""
echo "=== Step 6: annotate --level 3 ==="
python "$HIER_SEG_DIR/annotate.py" \
    --frames-dir "$FRAMES_DIR" \
    --l3-frames-dir "$FRAMES_L3_DIR" \
    --output-dir "$ANN_DIR" \
    --level 3 \
    --api-base "$API_BASE" \
    --model "$MODEL" \
    --workers "$WORKERS"

echo ""
echo "=========================================="
echo " Pipeline 完成!"
echo " Annotations → $ANN_DIR/"
echo " Frames L1   → $FRAMES_DIR/"
echo " Frames L3   → $FRAMES_L3_DIR/"
echo "=========================================="
