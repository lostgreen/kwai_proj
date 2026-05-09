#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FAMILY="qwen2_5_vl"
MODEL_SIZE="3b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen2.5-VL-3B-Instruct}"

source "${SCRIPT_DIR}/../../recipes/teacher_train_ema_grpo.sh"
