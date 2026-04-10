#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_pipeline_gemini.sh — 3-level annotation via Gemini native API
#
# Steps:
#   1. Extract 1fps frames (for visualization in data_visualization/)
#   2. Full video → Gemini → L1+L2+L3 in one call
#
# Usage:
#   tmux new -s anno_gemini
#   bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline_gemini.sh
#
# Test mode:
#   LIMIT=3 bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline_gemini.sh
#
# Skip frame extraction (if already done):
#   SKIP_FRAMES=1 bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline_gemini.sh
#
# With native API key instead of Vertex AI:
#   AUTH_MODE=apikey GEMINI_API_KEY=xxx bash proxy_data/youcook2_seg/hier_seg_annotation/run_pipeline_gemini.sh
#
# All steps are idempotent: already-completed clips are skipped.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail
export PYTHONUNBUFFERED=1

# ── Config ──────────────────────────────────────────────────────────
SCRIPT_DIR="proxy_data/youcook2_seg/hier_seg_annotation"
DATA_ROOT="/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1"

JSONL="${JSONL:-/home/xuboshen/zgw/EasyR1/proxy_data/data_curation/results/et_instruct_164k/screen_keep.jsonl}"
MODEL="${MODEL:-gemini-2.5-pro}"
FPS="${FPS:-2}"
LIMIT="${LIMIT:-20}"
WORKERS="${WORKERS:-4}"
FRAME_WORKERS="${FRAME_WORKERS:-8}"
SKIP_FRAMES="${SKIP_FRAMES:-0}"

# Auth: "vertex" (default, uses credential JSON) or "apikey" (uses GEMINI_API_KEY)
AUTH_MODE="${AUTH_MODE:-vertex}"
CREDENTIAL_JSON="${CREDENTIAL_JSON:-/home/liuxiaokun/projects/gemini-api/chatgpt-client/keling-ylab-gemini-1038ec8509a2.json}"

# Output directories
FRAMES_DIR="${DATA_ROOT}/frames"
OUTPUT_DIR="${DATA_ROOT}/annotations"

LOG_DIR="${DATA_ROOT}/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR" "$FRAMES_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/gemini_native_${TIMESTAMP}.log"

# Build --limit flag
LIMIT_FLAG=""
if [[ "$LIMIT" -gt 0 ]]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

# Build auth flags
AUTH_FLAGS=""
if [[ "$AUTH_MODE" == "vertex" ]]; then
    AUTH_FLAGS="--credential-json $CREDENTIAL_JSON"
elif [[ "$AUTH_MODE" == "apikey" ]]; then
    AUTH_FLAGS="--api-key ${GEMINI_API_KEY:-}"
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
    fi
}

log "========== GEMINI NATIVE PIPELINE CONFIG =========="
log "JSONL:         $JSONL"
log "MODEL:         $MODEL"
log "FPS:           $FPS"
log "WORKERS:       $WORKERS"
log "FRAME_WORKERS: $FRAME_WORKERS"
log "LIMIT:         ${LIMIT:-0 (all)}"
log "AUTH:          $AUTH_MODE"
log "FRAMES_DIR:    $FRAMES_DIR"
log "OUTPUT_DIR:    $OUTPUT_DIR"
log "DATA_ROOT:     $DATA_ROOT"

# =====================================================================
# STEP 1: Extract 1fps frames (for visualization)
# =====================================================================
if [[ "$SKIP_FRAMES" == "0" ]]; then
    log ""
    log ">>>>>>>>>> STEP 1: EXTRACT 1fps FRAMES <<<<<<<<<<"

    run_step "S1_EXTRACT_FRAMES" \
        python "$SCRIPT_DIR/extract_frames.py" \
            --jsonl "$JSONL" \
            --output-dir "$FRAMES_DIR" \
            --fps 1 --workers "$FRAME_WORKERS" \
            $LIMIT_FLAG
else
    log ""
    log ">>>>>>>>>> STEP 1: SKIP (SKIP_FRAMES=1) <<<<<<<<<<"
fi

# =====================================================================
# STEP 2: Full video → L1+L2+L3 annotation (Gemini native)
# =====================================================================
log ""
log ">>>>>>>>>> STEP 2: Gemini Native 3-Level Annotation <<<<<<<<<<"

CMD="python $SCRIPT_DIR/annotate_gemini_native.py \
    --data-path $JSONL \
    --output-dir $OUTPUT_DIR \
    --model $MODEL \
    --fps $FPS \
    --workers $WORKERS \
    $AUTH_FLAGS \
    $LIMIT_FLAG"

log "CMD: $CMD"

if eval "$CMD" 2>&1 | tee -a "$LOG_FILE"; then
    log "========== ANNOTATION DONE =========="
else
    log "========== ANNOTATION FAILED (exit $?) =========="
    log "Check log: $LOG_FILE"
fi

# ── Summary ─────────────────────────────────────────────────────────
log ""
log "========== PIPELINE COMPLETE =========="
log "Frames:      $FRAMES_DIR/"
log "Annotations: $OUTPUT_DIR/"
log "Full log:    $LOG_FILE"

# Count results
TOTAL=$(ls "$OUTPUT_DIR/"*.json 2>/dev/null | wc -l | tr -d ' ')
HAS_L3=$(grep -l '"level3"' "$OUTPUT_DIR/"*.json 2>/dev/null | wc -l | tr -d ' ')
SKIPPED=$(grep -l '"skip": true' "$OUTPUT_DIR/"*.json 2>/dev/null | wc -l | tr -d ' ')
log "Total: ${TOTAL} annotations, ${HAS_L3} with L3, ${SKIPPED} skipped (feasibility)"
