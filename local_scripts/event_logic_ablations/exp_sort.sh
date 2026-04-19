#!/usr/bin/env bash
# =============================================================
# exp_sort.sh — Event Logic Ablation: Sort (单任务)
#
# 多任务混合: TG + MCQ (base) + Event Logic (sort only)
# Reward: mixed_proxy_reward (MCQ → choice, TG → tIoU, EL → jigsaw displacement)
#
# 用法:
#   bash local_scripts/event_logic_ablations/exp_sort.sh
#   MAX_STEPS=30 bash local_scripts/event_logic_ablations/exp_sort.sh
# =============================================================
set -euo pipefail

# ---- VLM 管线输出目录 ----
_EL_VLM_DATA="${EL_VLM_DATA:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/event_logic}"

# ---- 实验配置 ----
export EXP_NAME="${EXP_NAME:-el_ablation_sort}"

# ---- 启用的任务 + 数据量 ----
export TASKS="${TASKS:-tg mcq event_logic}"
export EL_TRAIN="${EL_TRAIN:-${_EL_VLM_DATA}/train_sort.jsonl}"
export EL_VAL_SOURCE="${EL_VAL_SOURCE:-${_EL_VLM_DATA}/val_sort.jsonl}"
export EL_TARGET="${EL_TARGET:-5000}"
export VAL_EL_N="${VAL_EL_N:-150}"

# ---- Checkpoint 单独目录 ----
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/event_logic/ablations}"

# ---- sort 输出空间较大，1.0 也合适 ----
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"

# ---- 启动 (multi_task_common + 数据构建 + 训练) ----
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../run_multi_task.sh"
