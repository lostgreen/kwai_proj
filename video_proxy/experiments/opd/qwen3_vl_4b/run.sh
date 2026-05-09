#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FAMILY="qwen3_vl"
MODEL_SIZE="4b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
EXP_NAME="${EXP_NAME:-qwen3_vl_4b_opd}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_teacher_ema_grpo/global_step_200/actor/huggingface}"
ROLLOUT_BS="${ROLLOUT_BS:-64}"
GLOBAL_BS="${GLOBAL_BS:-64}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
SAVE_FREQ="${SAVE_FREQ:-50}"
SAVE_LIMIT="${SAVE_LIMIT:--1}"

source "${SCRIPT_DIR}/../../../training/recipes/opd_train.sh"
