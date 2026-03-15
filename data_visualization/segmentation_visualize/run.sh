#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8890}"

python "${ROOT_DIR}/server.py" --host "${HOST}" --port "${PORT}" --static-dir "${ROOT_DIR}"
