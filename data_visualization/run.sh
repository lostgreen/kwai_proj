#!/bin/bash
# Unified data visualization server
# Usage:
#   ./run.sh                                        # empty UI on :8787
#   ./run.sh --data-path /path/to/annotations      # pre-load seg data
#   ./run.sh --caption-pairs /path/cap_pairs.jsonl [--manifest /path/manifest.jsonl]
#   ./run.sh --mcq-data /path/mcq.jsonl
#   ./run.sh --port 9000 --data-path ... --mcq-data ...  # multi-load
set -euo pipefail
cd "$(dirname "$0")"
exec python server.py --port "${PORT:-8787}" "$@"
