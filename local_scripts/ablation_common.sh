#!/usr/bin/env bash
# Shared conservative defaults for ablation runs.

ABLATION_MODEL_PATH="${ABLATION_MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
ABLATION_CHECKPOINT_ROOT="${ABLATION_CHECKPOINT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations}"

if [[ "${ALLOW_ABLATION_MODEL_OVERRIDE:-false}" =~ ^(true|1|yes)$ ]]; then
    export MODEL_PATH="${MODEL_PATH:-${ABLATION_MODEL_PATH}}"
else
    export MODEL_PATH="${ABLATION_MODEL_PATH}"
fi

if [[ "${ALLOW_ABLATION_HPARAM_OVERRIDE:-false}" =~ ^(true|1|yes)$ ]]; then
    export LR="${LR:-5e-7}"
    export KL_COEF="${KL_COEF:-0.01}"
    export ENTROPY_COEFF="${ENTROPY_COEFF:-0.005}"
else
    export LR="5e-7"
    export KL_COEF="0.01"
    export ENTROPY_COEFF="0.005"
fi

if [[ "${ALLOW_ABLATION_CHECKPOINT_OVERRIDE:-false}" =~ ^(true|1|yes)$ ]]; then
    export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ABLATION_CHECKPOINT_ROOT}}"
else
    export CHECKPOINT_ROOT="${ABLATION_CHECKPOINT_ROOT}"
fi
