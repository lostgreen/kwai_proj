#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_event_logic_vlm.sh — VLM-powered Event Logic data construction
#
# Two-step pipeline:
#   Step 1: Cut atomic L2/L3 clips from annotations (prepare_all_clips.py)
#   Step 2: Call LLM Task Architect to design MCQ/sort questions
#
# Task types:
#   1. predict_next  — Predict the next step (MCQ)
#   2. fill_blank    — Fill-in-the-blank (MCQ)
#   3. sort          — Sequence sorting (digit-sequence)
#
# Usage:
#   # Full pipeline (clip + question generation)
#   bash proxy_data/youcook2_seg/event_logic/run_event_logic_vlm.sh
#
#   # Test mode (5 annotations, predict_next only)
#   LIMIT=5 TASKS="predict_next" bash proxy_data/youcook2_seg/event_logic/run_event_logic_vlm.sh
#
#   # Skip clip cutting (already done), only run question generation
#   SKIP_CLIPS=1 bash proxy_data/youcook2_seg/event_logic/run_event_logic_vlm.sh
#
#   # Dry-run (build script texts only, no API calls, no clip cutting)
#   DRY_RUN=1 LIMIT=10 bash proxy_data/youcook2_seg/event_logic/run_event_logic_vlm.sh
#
# Resume: cached LLM responses are stored in $OUTPUT_DIR/cache/.
#         Re-running the same command will skip already-cached annotations.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

export PYTHONUNBUFFERED=1

# ── Config ──────────────────────────────────────────────────────────
SCRIPT_DIR="proxy_data/youcook2_seg/event_logic"
CLIP_SCRIPT="proxy_data/youcook2_seg/prepare_all_clips.py"

# Data root (same as run_pipeline.sh)
DATA_ROOT="${DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1}"

# Derived paths
ANN_DIR="${ANN_DIR:-${DATA_ROOT}/annotations}"
CLIP_DIR="${CLIP_DIR:-${DATA_ROOT}/clips}"
SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/event_logic}"

# LLM config
API_BASE="${API_BASE:-https://api.novita.ai/v3/openai}"
MODEL="${MODEL:-pa/gmn-2.5-fls}"
TEMPERATURE="${TEMPERATURE:-0.7}"
WORKERS="${WORKERS:-8}"

# Task selection (space-separated: predict_next fill_blank sort)
TASKS="${TASKS:-predict_next fill_blank sort}"

# Data budget
TRAIN_BUDGET="${TRAIN_BUDGET:--1}"
VAL_COUNT="${VAL_COUNT:-150}"
SEED="${SEED:-42}"

# Processing limits
LIMIT="${LIMIT:-0}"
DRY_RUN="${DRY_RUN:-0}"
COMPLETE_ONLY="${COMPLETE_ONLY:-1}"
SKIP_CLIPS="${SKIP_CLIPS:-0}"

# Clip cutting config
CLIP_LEVELS="${CLIP_LEVELS:-L2 L3}"
CLIP_FPS="${CLIP_FPS:-2}"
CLIP_WORKERS="${CLIP_WORKERS:-8}"

# ── Logging ─────────────────────────────────────────────────────────
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/event_logic_vlm_${TIMESTAMP}.log"

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

# ── Print config ────────────────────────────────────────────────────
log "========== EVENT LOGIC VLM PIPELINE CONFIG =========="
log "DATA_ROOT:     $DATA_ROOT"
log "ANN_DIR:       $ANN_DIR"
log "CLIP_DIR:      $CLIP_DIR"
log "OUTPUT_DIR:    $OUTPUT_DIR"
log "MODEL:         $MODEL"
log "API_BASE:      $API_BASE"
log "WORKERS:       $WORKERS"
log "TEMPERATURE:   $TEMPERATURE"
log "TASKS:         $TASKS"
log "TRAIN_BUDGET:  $TRAIN_BUDGET"
log "VAL_COUNT:     $VAL_COUNT"
log "LIMIT:         ${LIMIT:-0 (all)}"
log "DRY_RUN:       $DRY_RUN"
log "SKIP_CLIPS:    $SKIP_CLIPS"
log "COMPLETE_ONLY: $COMPLETE_ONLY"
log "LOG_FILE:      $LOG_FILE"

# =====================================================================
# STEP 1: Cut atomic L2/L3 clips (skip if SKIP_CLIPS=1 or DRY_RUN=1)
# =====================================================================
if [[ "$SKIP_CLIPS" -eq 0 && "$DRY_RUN" -eq 0 ]]; then
    log ""
    log ">>>>>>>>>> STEP 1: CUT ATOMIC CLIPS (${CLIP_LEVELS}) <<<<<<<<<<"

    CLIP_FLAGS="--complete-only --levels $CLIP_LEVELS --l2l3-fps $CLIP_FPS --workers $CLIP_WORKERS"
    if [[ -n "$SOURCE_VIDEO_DIR" ]]; then
        CLIP_FLAGS="$CLIP_FLAGS --source-video-dir $SOURCE_VIDEO_DIR"
    fi

    run_step "S1_CUT_CLIPS" \
        python "$CLIP_SCRIPT" \
            --annotation-dir "$ANN_DIR" \
            --output-dir "$CLIP_DIR" \
            $CLIP_FLAGS
else
    log ""
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log ">>>>>>>>>> STEP 1: SKIP (dry-run mode) <<<<<<<<<<"
    else
        log ">>>>>>>>>> STEP 1: SKIP (SKIP_CLIPS=1) <<<<<<<<<<"
    fi
fi

# =====================================================================
# STEP 2: Build event logic questions via LLM Task Architect
# =====================================================================
log ""
log ">>>>>>>>>> STEP 2: BUILD EVENT LOGIC DATA <<<<<<<<<<"

FLAGS=""
if [[ "$LIMIT" -gt 0 ]]; then
    FLAGS="$FLAGS --limit $LIMIT"
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
    FLAGS="$FLAGS --dry-run"
fi
if [[ "$COMPLETE_ONLY" -eq 1 ]]; then
    FLAGS="$FLAGS --complete-only"
fi
if [[ "$TRAIN_BUDGET" -gt 0 ]]; then
    FLAGS="$FLAGS --train-budget $TRAIN_BUDGET"
fi

run_step "S2_EVENT_LOGIC_VLM" \
    python "$SCRIPT_DIR/build_event_logic_vlm.py" \
        --annotation-dir "$ANN_DIR" \
        --clip-dir "$CLIP_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --api-base "$API_BASE" \
        --model "$MODEL" \
        --temperature "$TEMPERATURE" \
        --workers "$WORKERS" \
        --tasks $TASKS \
        --val-count "$VAL_COUNT" \
        --cache-dir "$OUTPUT_DIR/cache" \
        --seed "$SEED" \
        $FLAGS

# ── Summary ─────────────────────────────────────────────────────────
log ""
log "========== PIPELINE COMPLETE =========="
if [[ -f "$OUTPUT_DIR/train.jsonl" ]]; then
    TRAIN_N=$(wc -l < "$OUTPUT_DIR/train.jsonl" | tr -d ' ')
    VAL_N=$(wc -l < "$OUTPUT_DIR/val.jsonl" | tr -d ' ')
    log "Output: $TRAIN_N train + $VAL_N val records"
    log "  train.jsonl: $OUTPUT_DIR/train.jsonl"
    log "  val.jsonl:   $OUTPUT_DIR/val.jsonl"
    log "  stats.json:  $OUTPUT_DIR/stats.json"
    log "  cache:       $OUTPUT_DIR/cache/"
fi

if [[ -d "$CLIP_DIR/L2" ]]; then
    L2_N=$(ls "$CLIP_DIR/L2/"*.mp4 2>/dev/null | wc -l | tr -d ' ')
    L3_N=$(ls "$CLIP_DIR/L3/"*.mp4 2>/dev/null | wc -l | tr -d ' ')
    log "Clips: $L2_N L2 + $L3_N L3"
fi

log "Full log: $LOG_FILE"
