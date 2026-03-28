#!/usr/bin/env bash
# run_build.sh — Build Event Logic proxy training dataset.
#
# Pipeline:
#   1. build_text_option_proxy.py  → raw proxy JSONL (add + replace + sort, CoT built-in)
#   2. filter_bad_videos.py        → filter unreadable / low-frame videos via decord
#
# Usage (on server):
#   cd /path/to/EasyR1
#   bash proxy_data/event_logic/run_build.sh
#
# Environment variables (override defaults as needed):
#   ANNOTATIONS      path to youcookii_annotations_trainval.json
#   EVENT_CLIPS_ROOT root directory of pre-extracted event clip mp4 files
#   OUTPUT_DIR       destination for generated JSONL files
#   WORKERS          parallel workers for video validation (default: 16)
#   SEED             random seed (default: 42)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ANNOTATIONS="${ANNOTATIONS:-${SCRIPT_DIR}/../youcookii_annotations_trainval.json}"
EVENT_CLIPS_ROOT="${EVENT_CLIPS_ROOT:-/m2v_intern/xuboshen/zgw/data/youcook2_event_clips}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/data}"
WORKERS="${WORKERS:-16}"
SEED="${SEED:-42}"

# L2 annotation pipeline paths
L2_ANNOTATION_DIR="${L2_ANNOTATION_DIR:-/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations}"
L2_CLIPS_DIR="${L2_CLIPS_DIR:-/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/clips/L2}"
L2_FRAMES_DIR="${L2_FRAMES_DIR:-/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/frames}"
L2_MODEL="${L2_MODEL:-qwen/qwen2.5-vl-72b-instruct}"
L2_API_BASE="${L2_API_BASE:-https://api.novita.ai/v3/openai}"
L2_CONFIDENCE="${L2_CONFIDENCE:-0.75}"
L2_FILTER_WORKERS="${L2_FILTER_WORKERS:-4}"

mkdir -p "$OUTPUT_DIR"

RAW_TRAIN="${OUTPUT_DIR}/proxy_train_text_options.jsonl"
CLEAN_TRAIN="${OUTPUT_DIR}/proxy_train_text_options_clean.jsonl"

# ── Step 1: Build raw proxy data ──────────────────────────────────────────────
echo "========================================"
echo "Step 1: Building proxy data (add / replace / sort)"
echo "========================================"
python "$SCRIPT_DIR/build_text_option_proxy.py" \
    --annotations      "$ANNOTATIONS" \
    --output           "$RAW_TRAIN" \
    --event-clips-root "$EVENT_CLIPS_ROOT" \
    --add-per-video    1 \
    --replace-per-video 1 \
    --sort-per-video   1 \
    --min-events       4 \
    --min-context      2 \
    --max-context      4 \
    --replace-seq-len  5 \
    --sort-seq-len     5 \
    --seed             "$SEED" \
    --shuffle

echo ""

# ── Step 2: Filter bad videos ─────────────────────────────────────────────────
echo "========================================"
echo "Step 2: Filtering unreadable / low-frame videos"
echo "========================================"
python "$SCRIPT_DIR/filter_bad_videos.py" \
    --input   "$RAW_TRAIN" \
    --output  "$CLEAN_TRAIN" \
    --workers "$WORKERS" \
    --min-frames 4

echo ""
echo "========================================"
echo "Done! Output directory: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "(no jsonl files found)"
echo ""
echo "EasyR1-ready dataset:"
echo "  $CLEAN_TRAIN"
echo "========================================"

echo ""
echo "========================================"
echo "Step 3: Building L2 Event Logic (add / replace / sort from L2 annotations)"
echo "========================================"

L2_RAW="${OUTPUT_DIR}/l2_event_logic_raw.jsonl"
L2_FILTERED="${OUTPUT_DIR}/l2_event_logic_filtered.jsonl"

python "$SCRIPT_DIR/build_l2_event_logic.py" \
    --annotation-dir  "$L2_ANNOTATION_DIR" \
    --clips-dir        "$L2_CLIPS_DIR" \
    --frames-dir       "$L2_FRAMES_DIR" \
    --output           "$L2_RAW" \
    --add-per-video    2 \
    --replace-per-video 2 \
    --sort-per-video   1 \
    --min-events       4 \
    --min-context      2 \
    --max-context      4 \
    --replace-seq-len  5 \
    --sort-seq-len     5 \
    --seed             "$SEED" \
    --shuffle

echo ""
echo "========================================"
echo "Step 4: AI causality filtering for L2 Event Logic"
echo "========================================"

python "$SCRIPT_DIR/build_l2_event_logic.py" \
    --annotation-dir  "$L2_ANNOTATION_DIR" \
    --clips-dir        "$L2_CLIPS_DIR" \
    --frames-dir       "$L2_FRAMES_DIR" \
    --output           "$L2_FILTERED" \
    --add-per-video    2 \
    --replace-per-video 2 \
    --sort-per-video   1 \
    --min-events       4 \
    --min-context      2 \
    --max-context      4 \
    --replace-seq-len  5 \
    --sort-seq-len     5 \
    --filter \
    --api-base         "$L2_API_BASE" \
    --model            "$L2_MODEL" \
    --confidence-threshold "$L2_CONFIDENCE" \
    --filter-workers   "$L2_FILTER_WORKERS" \
    --frames-per-event 3 \
    --seed             "$SEED" \
    --shuffle

echo ""
echo "========================================"
echo "L2 Event Logic done!"
echo "  Raw:      $L2_RAW"
echo "  Filtered: $L2_FILTERED"
echo "========================================"
