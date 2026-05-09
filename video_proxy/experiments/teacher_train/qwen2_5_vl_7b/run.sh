#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FAMILY="qwen2_5_vl"
MODEL_SIZE="7b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen2.5-VL-7B-Instruct}"
EXP_NAME="${EXP_NAME:-qwen2_5_vl_7b_teacher_ema_grpo}"

source "${SCRIPT_DIR}/../../../training/recipes/teacher_train_ema_grpo.sh"
