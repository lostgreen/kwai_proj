#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_pipeline.sh — Full annotation pipeline (Stage 1 + Stage 2)
#
# Stage 1: Classify (paradigm + domain + feasibility filtering)
# Stage 2: Annotate (L1+L2 merged + L3 leaf-node routing)
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
DATA_ROOT="/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation"

JSONL="${JSONL:-/home/xuboshen/zgw/EasyR1/proxy_data/data_curation/results/merged/sampled/sampled_1000.jsonl}"
MODEL="${MODEL:-pa/gemini-3.1-pro-preview}"
WORKERS="${WORKERS:-4}"
LIMIT="${LIMIT:-0}"

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
# STAGE 1: Classification (paradigm + domain + feasibility)
# =====================================================================
log ""
log ">>>>>>>>>> STAGE 1: CLASSIFICATION <<<<<<<<<<"

# ── Step 1.1: Extract 64 frames for classification ───────────────────
run_step "S1.1_EXTRACT_CLASSIFY_FRAMES" \
    python "$SCRIPT_DIR/extract_frames.py" \
        --jsonl "$JSONL" \
        --output-dir "$DATA_ROOT/frames_stage1" \
        --fps 1 --max-frames 64 --workers "$WORKERS" \
        $LIMIT_FLAG

# ── Step 1.2: VLM classification + feasibility filtering ─────────────
run_step "S1.2_CLASSIFY" \
    python "$SCRIPT_DIR/stage1_classify.py" \
        --jsonl "$JSONL" \
        --frames-dir "$DATA_ROOT/frames_stage1" \
        --output-dir "$DATA_ROOT/stage1_output" \
        --model "$MODEL" --workers "$WORKERS" \
        $LIMIT_FLAG

# Use classify_keep.jsonl as input for Stage 2
STAGE2_JSONL="$DATA_ROOT/stage1_output/classify_keep.jsonl"
if [[ ! -f "$STAGE2_JSONL" ]]; then
    log "ERROR: classify_keep.jsonl not found at $STAGE2_JSONL"
    log "Stage 1 may have failed. Falling back to original JSONL."
    STAGE2_JSONL="$JSONL"
fi
STAGE2_COUNT=$(wc -l < "$STAGE2_JSONL" | tr -d ' ')
log "Stage 2 input: $STAGE2_COUNT clips from $STAGE2_JSONL"

# =====================================================================
# STAGE 2: Hierarchical Annotation (L1+L2 merged + L3)
# =====================================================================
log ""
log ">>>>>>>>>> STAGE 2: ANNOTATION <<<<<<<<<<"

# ── Step 2.1: Extract 1fps frames for annotation ─────────────────────
run_step "S2.1_EXTRACT_ANNO_FRAMES" \
    python "$SCRIPT_DIR/extract_frames.py" \
        --jsonl "$STAGE2_JSONL" \
        --output-dir "$DATA_ROOT/frames" \
        --fps 1 --workers "$WORKERS" \
        $LIMIT_FLAG

# ── Step 2.2: Classify archetype + Merged L1+L2 annotation ──────────
run_step "S2.2_CLASSIFY_AND_MERGED" \
    python "$SCRIPT_DIR/annotate.py" \
        --jsonl "$STAGE2_JSONL" \
        --frames-dir "$DATA_ROOT/frames" \
        --output-dir "$DATA_ROOT/annotations" \
        --level merged \
        --classify-frames 64 \
        --model "$MODEL" --workers "$WORKERS" \
        $LIMIT_FLAG

# ── Step 2.3: Extract L3 frames (leaf-node routing) ──────────────────
run_step "S2.3_EXTRACT_FRAMES_L3" \
    python "$SCRIPT_DIR/extract_frames.py" \
        --annotation-dir "$DATA_ROOT/annotations" \
        --output-dir "$DATA_ROOT/frames_l3" \
        --fps 2 --workers "$WORKERS" \
        $LIMIT_FLAG

# ── Step 2.4: L3 annotation (leaf-node routing) ──────────────────────
run_step "S2.4_L3_ANNOTATION" \
    python "$SCRIPT_DIR/annotate.py" \
        --jsonl "$STAGE2_JSONL" \
        --frames-dir "$DATA_ROOT/frames" \
        --l3-frames-dir "$DATA_ROOT/frames_l3" \
        --output-dir "$DATA_ROOT/annotations" \
        --level 3 \
        --model "$MODEL" --workers "$WORKERS" \
        $LIMIT_FLAG

# ── Summary ─────────────────────────────────────────────────────────
log ""
log "========== PIPELINE COMPLETE =========="
log "Stage 1 output: $DATA_ROOT/stage1_output/"
log "Annotations:    $DATA_ROOT/annotations/"
log "Full log:       $LOG_FILE"

# Count results
if ls "$DATA_ROOT/stage1_output/classify_keep.jsonl" &>/dev/null; then
    S1_KEEP=$(wc -l < "$DATA_ROOT/stage1_output/classify_keep.jsonl" | tr -d ' ')
    S1_REJECT=$(wc -l < "$DATA_ROOT/stage1_output/classify_reject.jsonl" 2>/dev/null | tr -d ' ' || echo 0)
    log "Stage 1: ${S1_KEEP} keep, ${S1_REJECT} reject"
fi

TOTAL=$(ls "$DATA_ROOT/annotations/"*.json 2>/dev/null | wc -l | tr -d ' ')
HAS_L3=$(grep -l '"level3"' "$DATA_ROOT/annotations/"*.json 2>/dev/null | wc -l | tr -d ' ')
log "Stage 2: ${TOTAL} annotations, ${HAS_L3} with L3"
