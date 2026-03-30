#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_pipeline.sh — One-shot annotation pipeline for tmux overnight run
#
# Usage:
#   tmux new -s anno
#   bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline.sh
#   # Ctrl-B D to detach
#
# All steps are idempotent: already-completed clips are skipped.
# Any per-clip crash is caught & logged, pipeline continues.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# Force unbuffered Python output (critical for tee piping)
export PYTHONUNBUFFERED=1

# ── Config ──────────────────────────────────────────────────────────
SCRIPT_DIR="proxy_data/youcook2_seg/hier_seg_annotation"
DATA_ROOT="/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation"

JSONL="${JSONL:-/home/xuboshen/zgw/EasyR1/proxy_data/data_curation/results/merged/sampled/sampled_1000.jsonl}"
MODEL="${MODEL:-pa/gemini-3.1-pro-preview}"
WORKERS="${WORKERS:-4}"

LOG_DIR="${DATA_ROOT}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/pipeline_${TIMESTAMP}.log"

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

# ── Step 1: Extract 1fps frames (L1+L2) ────────────────────────────
run_step "STEP1_EXTRACT_FRAMES_L1" \
    python "$SCRIPT_DIR/extract_frames.py" \
        --jsonl "$JSONL" \
        --output-dir "$DATA_ROOT/frames" \
        --fps 1 --workers "$WORKERS"

# ── Step 2: Merged annotation (L1+L2+Topology+Criterion) ───────────
run_step "STEP2_MERGED_ANNOTATION" \
    python "$SCRIPT_DIR/annotate.py" \
        --jsonl "$JSONL" \
        --frames-dir "$DATA_ROOT/frames" \
        --output-dir "$DATA_ROOT/annotations" \
        --level merged \
        --model "$MODEL" --workers "$WORKERS"

# ── Step 3: Extract L3 frames (leaf-node routing) ──────────────────
run_step "STEP3_EXTRACT_FRAMES_L3" \
    python "$SCRIPT_DIR/extract_frames.py" \
        --annotation-dir "$DATA_ROOT/annotations" \
        --output-dir "$DATA_ROOT/frames_l3" \
        --fps 2 --workers "$WORKERS"

# ── Step 4: L3 annotation (leaf-node routing) ──────────────────────
run_step "STEP4_L3_ANNOTATION" \
    python "$SCRIPT_DIR/annotate.py" \
        --jsonl "$JSONL" \
        --frames-dir "$DATA_ROOT/frames" \
        --l3-frames-dir "$DATA_ROOT/frames_l3" \
        --output-dir "$DATA_ROOT/annotations" \
        --level 3 \
        --model "$MODEL" --workers "$WORKERS"

# ── Summary ─────────────────────────────────────────────────────────
log "========== PIPELINE COMPLETE =========="
log "Annotations: $DATA_ROOT/annotations/"
log "Full log:    $LOG_FILE"

# Count results
TOTAL=$(ls "$DATA_ROOT/annotations/"*.json 2>/dev/null | wc -l)
HAS_L3=$(grep -l '"level3"' "$DATA_ROOT/annotations/"*.json 2>/dev/null | wc -l)
log "Total annotations: $TOTAL, with L3: $HAS_L3"
