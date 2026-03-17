#!/usr/bin/env bash
# run_build.sh — Build training datasets from hierarchical annotations on the server.
#
# Usage (on server as xuboshen):
#   cd /home/xuboshen/zgw/EasyR1
#   bash proxy_data/youcook2_seg_annotation/run_build.sh
#
# Because L1 has 500 annotated clips but L2/L3 only have 100,
# --complete-only filters to the ~100 clips that have all 3 levels.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANNOTATION_DIR="/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations"
OUTPUT_DIR="/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/datasets"

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Building Level 1 dataset (complete-only)"
echo "========================================"
python "$SCRIPT_DIR/build_dataset.py" \
  --annotation-dir "$ANNOTATION_DIR" \
  --output "$OUTPUT_DIR/youcook2_hier_L1_train.jsonl" \
  --level 1 \
  --complete-only

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
echo "Building Level 3 dataset (complete-only)"
echo "========================================"
python "$SCRIPT_DIR/build_dataset.py" \
  --annotation-dir "$ANNOTATION_DIR" \
  --output "$OUTPUT_DIR/youcook2_hier_L3_train.jsonl" \
  --level 3 \
  --l3-min-actions 3 \
  --complete-only

echo ""
echo "========================================"
echo "Done! Output directory: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "(no jsonl files found)"

echo ""
echo "========================================"
echo "To visualize a dataset, run one of the following from the repo root:"
echo ""
echo "  # Visualize L1 training data"
echo "  DATA_PATH=$OUTPUT_DIR/youcook2_hier_L1_train.jsonl \\"
echo "    PORT=8891 bash data_visualization/segmentation_visualize/run.sh"
echo ""
echo "  # Visualize L2 training data"
echo "  DATA_PATH=$OUTPUT_DIR/youcook2_hier_L2_train.jsonl \\"
echo "    PORT=8892 bash data_visualization/segmentation_visualize/run.sh"
echo ""
echo "  # Visualize L3 training data (recommended first check)"
echo "  DATA_PATH=$OUTPUT_DIR/youcook2_hier_L3_train.jsonl \\"
echo "    PORT=8893 bash data_visualization/segmentation_visualize/run.sh"
echo ""
echo "  # Or visualize raw annotation JSONs (all 3 levels together)"
echo "  ANNOTATION_DIR=$ANNOTATION_DIR \\"
echo "    PORT=8890 bash data_visualization/segmentation_visualize/run.sh"
echo "========================================"
