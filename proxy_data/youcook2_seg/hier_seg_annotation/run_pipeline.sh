#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_pipeline.sh — Scene-first hierarchical annotation pipeline
#
# Steps:
#   1. Extract 1fps frames
#   1.5. Scene detection (PySceneDetect → hard scene anchors)
#   2. Annotate (scene-first, two-pass):
#      Pass 1: Merge-decision + event caption + domain (1fps)
#      Pass 2: Per-event L3 sub-split (1fps, only event frames)
#      + L1 phase aggregation — all within annotate.py
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
MODEL="${MODEL:-pa/gmn-2.5-pr}"
WORKERS="${WORKERS:-8}"
LIMIT="${LIMIT:-5}"
EXTRACT_FPS="${EXTRACT_FPS:-1}"

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
log "JSONL:       $JSONL"
log "MODEL:       $MODEL"
log "WORKERS:     $WORKERS"
log "LIMIT:       ${LIMIT:-0 (all)}"
log "EXTRACT_FPS: $EXTRACT_FPS"
log "DATA_ROOT:   $DATA_ROOT"

# =====================================================================
# STEP 1: Extract frames at 2fps
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
# STEP 1.5: Scene Detection (PySceneDetect → hard scene anchors)
# =====================================================================
log ""
log ">>>>>>>>>> STEP 1.5: SCENE DETECTION <<<<<<<<<<"
log "    Scenes are HARD ANCHORS — Pass 1 (1fps) merges scenes + caption; Pass 2 (1fps) per-event L3"

SCENE_DETECTOR="${SCENE_DETECTOR:-content}"
SCENE_THRESHOLD="${SCENE_THRESHOLD:-27.0}"

run_step "S1_5_SCENE_DETECT" \
    python "$SCRIPT_DIR/detect_scenes.py" \
        --frames-dir "$DATA_ROOT/frames" \
        --detector "$SCENE_DETECTOR" \
        --threshold "$SCENE_THRESHOLD" \
        --workers "$WORKERS" \
        $LIMIT_FLAG

# =====================================================================
# STEP 2: Scene-first annotation (two-pass + L1 aggregation)
# =====================================================================
log ""
log ">>>>>>>>>> STEP 2: ANNOTATE (scene-first, two-pass) <<<<<<<<<<"

run_step "S2_ANNOTATE" \
    python "$SCRIPT_DIR/annotate.py" \
        --jsonl "$JSONL" \
        --frames-dir "$DATA_ROOT/frames" \
        --output-dir "$DATA_ROOT/annotations" \
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
