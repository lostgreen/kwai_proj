#!/bin/bash
# Unified data visualization server
# All parameters can be passed as CLI args or as environment variables:
#   DATA_PATH        → --data-path        (seg annotation dir / jsonl)
#   CAPTION_PAIRS    → --caption-pairs    (caption_pairs.jsonl)
#   MANIFEST         → --manifest         (aot_event_manifest.jsonl, paired with caption)
#   MCQ_DATA         → --mcq-data         (v2t/t2v/4way MCQ jsonl)
#   MAX_SAMPLES      → --max-samples
#   LAZY=1           → --lazy             (skip frame preload; load clips on demand)
#   PORT             → --port             (default 8890)
#
# Examples (env-var style, convenient for server use):
#   CAPTION_PAIRS=/path/caption_pairs.jsonl MANIFEST=/path/manifest.jsonl ./run.sh
#   DATA_PATH=/path/to/annotations ./run.sh
#   DATA_PATH=/path/to/annotations LAZY=1 ./run.sh   # fast startup, on-demand clip loading
#   MCQ_DATA=/path/v2t_train.jsonl ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

EXTRA_ARGS=()
[[ -n "${DATA_PATH:-}"     ]] && EXTRA_ARGS+=(--data-path     "$DATA_PATH")
[[ -n "${CAPTION_PAIRS:-}" ]] && EXTRA_ARGS+=(--caption-pairs "$CAPTION_PAIRS")
[[ -n "${MANIFEST:-}"      ]] && EXTRA_ARGS+=(--manifest      "$MANIFEST")
[[ -n "${MCQ_DATA:-}"      ]] && EXTRA_ARGS+=(--mcq-data      "$MCQ_DATA")
[[ -n "${MAX_SAMPLES:-}"   ]] && EXTRA_ARGS+=(--max-samples   "$MAX_SAMPLES")
[[ "${LAZY:-0}" == "1"     ]] && EXTRA_ARGS+=(--lazy)

exec python server.py --port "${PORT:-8890}" "${EXTRA_ARGS[@]}" "$@"
