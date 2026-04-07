#!/usr/bin/env bash
# run_demo.sh — End-to-end demo: sample → extract frames → annotate → build training data
#
# Usage (on server, from train/ directory):
#   bash proxy_data/youcook2_seg/grounding_seg/run_demo.sh
#
# Environment variables (all have defaults):
#   PER_SOURCE   — samples per source domain (default: 50)
#   WORKERS      — parallel workers for extraction/annotation (default: 4)
#   MODEL        — VLM model (default: pa/gmn-2.5-fl)
#   API_BASE     — VLM API endpoint (default: https://api.novita.ai/v3/openai)
#   SKIP_SAMPLE  — set "true" to reuse existing sampled JSONL
#   SKIP_FRAMES  — set "true" to skip frame extraction
#   SKIP_ANNOTATE — set "true" to skip VLM annotation
#   LIMIT        — max clips to annotate (default: 0 = all sampled)

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="proxy_data/youcook2_seg/grounding_seg"
HIER_SCRIPT_DIR="proxy_data/youcook2_seg/hier_seg_annotation"

DATA_ROOT="/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/grounding_seg"
FRAMES_DIR="${DATA_ROOT}/frames"
ANN_DIR="${DATA_ROOT}/annotations"
TRAIN_DIR="${DATA_ROOT}/train_data"
VIDEO_DIR="/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/videos"

# Input source
INPUT_JSONL="${INPUT_JSONL:-proxy_data/data_curation/results/merged/candidates.jsonl}"
SAMPLED_JSONL="${DATA_ROOT}/sampled_demo.jsonl"

# ─── Config ───────────────────────────────────────────────────────────────────
PER_SOURCE="${PER_SOURCE:-50}"
WORKERS="${WORKERS:-4}"
MODEL="${MODEL:-pa/gmn-2.5-fl}"
API_BASE="${API_BASE:-https://api.novita.ai/v3/openai}"
SKIP_SAMPLE="${SKIP_SAMPLE:-false}"
SKIP_FRAMES="${SKIP_FRAMES:-false}"
SKIP_ANNOTATE="${SKIP_ANNOTATE:-false}"
LIMIT="${LIMIT:-0}"

echo "============================================"
echo " Grounding+Seg Demo Pipeline"
echo "============================================"
echo " INPUT_JSONL:  $INPUT_JSONL"
echo " DATA_ROOT:    $DATA_ROOT"
echo " PER_SOURCE:   $PER_SOURCE"
echo " MODEL:        $MODEL"
echo " WORKERS:      $WORKERS"
echo " LIMIT:        $LIMIT"
echo "============================================"

# ─── Step 0: Balanced sampling ────────────────────────────────────────────────
if [ "$SKIP_SAMPLE" = "true" ] && [ -f "$SAMPLED_JSONL" ]; then
    echo ""
    echo ">>> Step 0: SKIP sampling (reusing $SAMPLED_JSONL)"
    N_SAMPLED=$(wc -l < "$SAMPLED_JSONL" | tr -d ' ')
    echo "    $N_SAMPLED records"
else
    echo ""
    echo ">>> Step 0: Balanced sampling ($PER_SOURCE per source)"
    mkdir -p "$DATA_ROOT"
    python "$SCRIPT_DIR/sample_for_demo.py" \
        --input "$INPUT_JSONL" \
        --output "$SAMPLED_JSONL" \
        --per-source "$PER_SOURCE" \
        --seed 42
fi

# ─── Step 1: Extract frames ──────────────────────────────────────────────────
if [ "$SKIP_FRAMES" = "true" ]; then
    echo ""
    echo ">>> Step 1: SKIP frame extraction"
else
    echo ""
    echo ">>> Step 1: Extracting frames (1fps)"
    python "$HIER_SCRIPT_DIR/extract_frames.py" \
        --jsonl "$SAMPLED_JSONL" \
        --output-dir "$FRAMES_DIR" \
        --fps 1 \
        --workers "$WORKERS"
fi

# ─── Step 2: VLM annotation ──────────────────────────────────────────────────
if [ "$SKIP_ANNOTATE" = "true" ]; then
    echo ""
    echo ">>> Step 2: SKIP VLM annotation"
else
    echo ""
    echo ">>> Step 2: VLM annotation (model=$MODEL)"

    ANNOTATE_ARGS=(
        --jsonl "$SAMPLED_JSONL"
        --frames-dir "$FRAMES_DIR"
        --output-dir "$ANN_DIR"
        --api-base "$API_BASE"
        --model "$MODEL"
        --max-frames-per-call 64
        --workers "$WORKERS"
    )
    if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
        ANNOTATE_ARGS+=(--limit "$LIMIT")
    fi

    python "$SCRIPT_DIR/annotate_gseg.py" "${ANNOTATE_ARGS[@]}"
fi

# ─── Step 3: Build training data ─────────────────────────────────────────────
echo ""
echo ">>> Step 3: Building training JSONL"
python "$SCRIPT_DIR/build_gseg_data.py" \
    --annotation-dir "$ANN_DIR" \
    --output-dir "$TRAIN_DIR" \
    --video-dir "$VIDEO_DIR" \
    --min-segments 2 \
    --max-segments 15

echo ""
echo "============================================"
echo " Done! Training data:"
echo "   $TRAIN_DIR/train.jsonl"
echo "   $TRAIN_DIR/val.jsonl"
echo "============================================"

# Quick stats
if [ -f "$TRAIN_DIR/train.jsonl" ]; then
    N_TRAIN=$(wc -l < "$TRAIN_DIR/train.jsonl" | tr -d ' ')
    N_VAL=$(wc -l < "$TRAIN_DIR/val.jsonl" | tr -d ' ')
    echo "   Train: $N_TRAIN  |  Val: $N_VAL"
fi
