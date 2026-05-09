#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "[compat] video pure EMA-GRPO moved to video_proxy/training/models/qwen3_vl_8b/teacher_train_ema_grpo.sh" >&2
source "${SCRIPT_DIR}/../models/qwen3_vl_8b/teacher_train_ema_grpo.sh"
