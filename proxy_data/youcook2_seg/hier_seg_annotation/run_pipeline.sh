#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_pipeline.sh — Unified annotation pipeline (v5)
#
# v5: Classification + L1/L2 annotation fused into a single VLM call.
#     Stage 1 (separate classification) removed.
#
# Steps:
#   1. Extract 1fps frames
#   2. Unified classify + L1/L2 annotation (single VLM call per clip)
#   3. Extract L3 frames (leaf-node routing)
#   4. L3 annotation
#
# Usage:
#   tmux new -s anno
#   bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline.sh
#   # Ctrl-B D to detach
#
# Test mode (process only N clips):
#   LIMIT=5 bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline.sh
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
log "JSONL:    $JSONL"
log "MODEL:    $MODEL"
log "WORKERS:  $WORKERS"
log "LIMIT:    ${LIMIT:-0 (all)}"
log "DATA_ROOT: $DATA_ROOT"

# =====================================================================
# STEP 1: Extract 1fps frames
# =====================================================================
log ""
log ">>>>>>>>>> STEP 1: EXTRACT FRAMES <<<<<<<<<<"

run_step "S1_EXTRACT_FRAMES" \
    python "$SCRIPT_DIR/extract_frames.py" \
        --jsonl "$JSONL" \
        --output-dir "$DATA_ROOT/frames" \
        --fps 1 --workers "$WORKERS" \
        $LIMIT_FLAG

# =====================================================================
# STEP 2: Unified classify + L1/L2 annotation (single VLM call)
# =====================================================================
log ""
log ">>>>>>>>>> STEP 2: UNIFIED CLASSIFY + ANNOTATE <<<<<<<<<<"

run_step "S2_UNIFIED_MERGED" \
    python "$SCRIPT_DIR/annotate.py" \
        --jsonl "$JSONL" \
        --frames-dir "$DATA_ROOT/frames" \
        --output-dir "$DATA_ROOT/annotations" \
        --level merged \
        --model "$MODEL" --workers "$WORKERS" \
        $LIMIT_FLAG

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
