#!/usr/bin/env bash
# Shared 8B defaults for task-composition ablations.

COMPOSITION_8B_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export PROJECT_NAME="${PROJECT_NAME:-EasyR1-task-composition-ablation-8b}"
export ABLATION_MODEL_PATH="${ABLATION_MODEL_PATH:-${ABLATION_8B_MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct}}"
export ABLATION_CHECKPOINT_ROOT="${ABLATION_CHECKPOINT_ROOT:-${ABLATION_8B_CHECKPOINT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_8b_lr5e-7_kl0p01_entropy0p005_ablations}}"
export ABLATION_4B_EXPERIMENTS_DIR="${ABLATION_4B_EXPERIMENTS_DIR:-${MULTI_TASK_DATA_ROOT:-${THREE_TASK_DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task}}/experiments}"

source "${COMPOSITION_8B_DIR}/common.sh"

export TP_SIZE="${TP_SIZE:-2}"
export ROLLOUT_BS="${ROLLOUT_BS:-32}"
export GLOBAL_BS="${GLOBAL_BS:-32}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
export FILLER_GPUS="${FILLER_GPUS:-0,1,2,3,4,6,7}"
