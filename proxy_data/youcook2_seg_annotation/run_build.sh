#!/usr/bin/env bash
# run_build.sh — Build training datasets from hierarchical annotations on the server.
#
# Usage (on server as xuboshen):
#   cd /home/xuboshen/zgw/EasyR1
#   bash proxy_data/youcook2_seg_annotation/run_build.sh
#
# Pipeline:
#   L1: build_dataset.py  → prepare_clips.py  (concat selected JPEG frames → synthetic mp4)
#   L2: build_dataset.py  → prepare_clips.py  (extract 128s windows, normalize timestamps)
#   L3: build_dataset.py  → prepare_clips.py  (extract event clips, normalize timestamps)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANNOTATION_DIR="/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations"
OUTPUT_DIR="/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/datasets"
CLIP_DIR="/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/clips"
WORKERS="${WORKERS:-8}"

mkdir -p "$OUTPUT_DIR" "$CLIP_DIR"

echo "========================================"
echo "Building Level 1 dataset (complete-only)"
echo "========================================"
python "$SCRIPT_DIR/build_dataset.py" \
  --annotation-dir "$ANNOTATION_DIR" \
  --output "$OUTPUT_DIR/youcook2_hier_L1_train.jsonl" \
  --level 1 \
  --max-frames 256 \
  --complete-only

echo ""
echo "========================================"
echo "Assembling L1 warped-frame clips"
echo "========================================"
python "$SCRIPT_DIR/prepare_clips.py" \
  --input   "$OUTPUT_DIR/youcook2_hier_L1_train.jsonl" \
  --output  "$OUTPUT_DIR/youcook2_hier_L1_train_clipped.jsonl" \
  --clip-dir "$CLIP_DIR/L1" \
  --workers "$WORKERS"

echo ""
echo "========================================"
echo "Building Level 2 dataset (complete-only)"
echo "========================================"
python "$SCRIPT_DIR/build_dataset.py" \
  --annotation-dir "$ANNOTATION_DIR" \
  --output "$OUTPUT_DIR/youcook2_hier_L2_train.jsonl" \
  --level 2 \
  --l2-window-size 128 \
  --l2-stride 64 \
  --l2-min-events 2 \
  --complete-only

echo ""
echo "========================================"
echo "Extracting L2 sub-clips & normalizing timestamps"
echo "========================================"
python "$SCRIPT_DIR/prepare_clips.py" \
  --input   "$OUTPUT_DIR/youcook2_hier_L2_train.jsonl" \
  --output  "$OUTPUT_DIR/youcook2_hier_L2_train_clipped.jsonl" \
  --clip-dir "$CLIP_DIR/L2" \
  --workers "$WORKERS"

echo ""
echo "========================================"
echo "Building Level 3 dataset (complete-only)"
echo "========================================"
python "$SCRIPT_DIR/build_dataset.py" \
  --annotation-dir "$ANNOTATION_DIR" \
  --output "$OUTPUT_DIR/youcook2_hier_L3_train.jsonl" \
  --level 3 \
  --l3-min-actions 3 \
  --l3-padding 5 \
  --l3-order both \
  --complete-only

echo ""
echo "========================================"
echo "Extracting L3 sub-clips & normalizing timestamps"
echo "========================================"
python "$SCRIPT_DIR/prepare_clips.py" \
  --input   "$OUTPUT_DIR/youcook2_hier_L3_train.jsonl" \
  --output  "$OUTPUT_DIR/youcook2_hier_L3_train_clipped.jsonl" \
  --clip-dir "$CLIP_DIR/L3" \
  --workers "$WORKERS"

echo ""
echo "========================================"
echo "Done! Output directory: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "(no jsonl files found)"
echo ""
echo "EasyR1-ready datasets:"
echo "  L1: $OUTPUT_DIR/youcook2_hier_L1_train_clipped.jsonl"
echo "  L2: $OUTPUT_DIR/youcook2_hier_L2_train_clipped.jsonl"
echo "  L3: $OUTPUT_DIR/youcook2_hier_L3_train_clipped.jsonl"
echo "========================================"
