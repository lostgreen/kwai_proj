#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_pipeline.sh — Hierarchical annotation pipeline (v7/v8)
#
# Steps (l2_first — default bottom-up):
#   1. Extract 1fps frames
#   2. L2-first dense captioning + L1 aggregation (bottom-up)
#   3. Extract L3 frames (leaf-node routing)
#   4. L3 annotation
#
# Steps (l2l3_first — 2fps, all-in-one L2+L3):
#   1. Extract 2fps frames
#   2. L2+L3 annotation + L1 aggregation (2 VLM calls, no L3 step)
#
# Usage:
#   tmux new -s anno
#   bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline.sh
#   # Ctrl-B D to detach
#
# Test mode (process only N clips):
#   LIMIT=5 bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline.sh
#
# L2L3 mode (2fps, one-shot L2+L3):
#   ANNO_LEVEL=l2l3_first bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline.sh
#
# Old top-down pipeline:
#   ANNO_LEVEL=merged bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline.sh
#
# All steps are idempotent: already-completed clips are skipped.
# Any per-clip crash is caught & logged, pipeline continues.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# Force unbuffered Python output (critical for tee piping)
export PYTHONUNBUFFERED=1

# ── Config ──────────────────────────────────────────────────────────
SCRIPT_DIR="proxy_data/youcook2_seg/hier_seg_annotation"
DATA_ROOT="/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1"

JSONL="${JSONL:-/home/xuboshen/zgw/EasyR1/proxy_data/data_curation/results/et_instruct_164k/screen_keep.jsonl}"
MODEL="${MODEL:-pa/gmn-2.5-fls}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-20}"
ANNO_LEVEL="${ANNO_LEVEL:-l2_first}"  # "l2l3_first" | "l2_first" (bottom-up) | "merged" (old top-down)

# FPS: 2fps for l2l3_first, 1fps otherwise
if [[ "$ANNO_LEVEL" == "l2l3_first" ]]; then
    EXTRACT_FPS="${EXTRACT_FPS:-2}"
else
    EXTRACT_FPS="${EXTRACT_FPS:-1}"
fi

LOG_DIR="${DATA_ROOT}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

# Build --limit flag (empty string when LIMIT=0 means "all")
LIMIT_FLAG=""
if [[ "$LIMIT" -gt 0 ]]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

# ── Helpers ─────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

run_step() {
    local step_name="$1"
    shift
    log "========== ${step_name} START =========="
    log "CMD: $*"
    if "$@" 2>&1 | tee -a "$LOG_FILE"; then
        log "========== ${step_name} DONE =========="
    else
        log "========== ${step_name} FAILED (exit $?) =========="
        log "Check log: $LOG_FILE"
        # Continue to next step instead of aborting
    fi
}

log "========== PIPELINE CONFIG =========="
log "JSONL:      $JSONL"
log "MODEL:      $MODEL"
log "WORKERS:    $WORKERS"
log "LIMIT:      ${LIMIT:-0 (all)}"
log "ANNO_LEVEL: $ANNO_LEVEL"
log "EXTRACT_FPS: $EXTRACT_FPS"
log "DATA_ROOT:  $DATA_ROOT"

# =====================================================================
# STEP 1: Extract frames (1fps or 2fps depending on ANNO_LEVEL)
# =====================================================================
log ""
log ">>>>>>>>>> STEP 1: EXTRACT FRAMES (${EXTRACT_FPS}fps) <<<<<<<<<<"

run_step "S1_EXTRACT_FRAMES" \
    python "$SCRIPT_DIR/extract_frames.py" \
        --jsonl "$JSONL" \
        --output-dir "$DATA_ROOT/frames" \
        --fps "$EXTRACT_FPS" --workers "$WORKERS" \
        $LIMIT_FLAG

# =====================================================================
# STEP 2: L2-first annotation + L1 aggregation (or old merged mode)
# =====================================================================
log ""
log ">>>>>>>>>> STEP 2: ANNOTATE (${ANNO_LEVEL}) <<<<<<<<<<"

run_step "S2_ANNOTATE" \
    python "$SCRIPT_DIR/annotate.py" \
        --jsonl "$JSONL" \
        --frames-dir "$DATA_ROOT/frames" \
        --output-dir "$DATA_ROOT/annotations" \
        --level "$ANNO_LEVEL" \
        --model "$MODEL" --workers "$WORKERS" \
        $LIMIT_FLAG

# =====================================================================
# STEP 3 & 4: L3 frames + annotation (skip for l2l3_first — L3 is inline)
# =====================================================================
if [[ "$ANNO_LEVEL" == "l2l3_first" ]]; then
    log ""
    log ">>>>>>>>>> STEPS 3-4 SKIPPED (L3 inline in l2l3_first mode) <<<<<<<<<<"
else
# =====================================================================
# STEP 3: Extract L3 frames (leaf-node routing)
# =====================================================================
log ""
log ">>>>>>>>>> STEP 3: EXTRACT L3 FRAMES <<<<<<<<<<"

run_step "S3_EXTRACT_FRAMES_L3" \
    python "$SCRIPT_DIR/extract_frames.py" \
        --annotation-dir "$DATA_ROOT/annotations" \
        --output-dir "$DATA_ROOT/frames_l3" \
        --fps 2 --workers "$WORKERS" \
        $LIMIT_FLAG

# =====================================================================
# STEP 4: L3 annotation (leaf-node routing)
# =====================================================================
log ""
log ">>>>>>>>>> STEP 4: L3 ANNOTATION <<<<<<<<<<"

run_step "S4_L3_ANNOTATION" \
    python "$SCRIPT_DIR/annotate.py" \
        --jsonl "$JSONL" \
        --frames-dir "$DATA_ROOT/frames" \
        --l3-frames-dir "$DATA_ROOT/frames_l3" \
        --output-dir "$DATA_ROOT/annotations" \
        --level 3 \
        --model "$MODEL" --workers "$WORKERS" \
        $LIMIT_FLAG
fi

# ── Summary ─────────────────────────────────────────────────────────
log ""
log "========== PIPELINE COMPLETE =========="
log "Annotations: $DATA_ROOT/annotations/"
log "Full log:    $LOG_FILE"

# Count results
TOTAL=$(ls "$DATA_ROOT/annotations/"*.json 2>/dev/null | wc -l | tr -d ' ')
HAS_L3=$(grep -l '"level3"' "$DATA_ROOT/annotations/"*.json 2>/dev/null | wc -l | tr -d ' ')
SKIPPED=$(grep -l '"skip": true' "$DATA_ROOT/annotations/"*.json 2>/dev/null | wc -l | tr -d ' ')
log "Total: ${TOTAL} annotations, ${HAS_L3} with L3, ${SKIPPED} skipped (feasibility)"
