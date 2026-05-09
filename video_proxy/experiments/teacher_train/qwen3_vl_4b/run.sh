#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FAMILY="qwen3_vl"
MODEL_SIZE="4b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
EXP_NAME="${EXP_NAME:-qwen3_vl_4b_teacher_ema_grpo}"

source "${SCRIPT_DIR}/../../../training/recipes/teacher_train_ema_grpo.sh"
