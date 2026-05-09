#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FAMILY="qwen2_5_vl"
MODEL_SIZE="7b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen2.5-VL-7B-Instruct}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen2_5_vl_7b_teacher_ema_grpo/global_step_200/actor/huggingface}"

source "${SCRIPT_DIR}/../../recipes/opd_train.sh"
