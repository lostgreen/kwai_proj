#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "[compat] single-teacher OPD moved to video_proxy/training/models/qwen3_vl_4b/opd_train.sh" >&2
source "${SCRIPT_DIR}/../models/qwen3_vl_4b/opd_train.sh"
