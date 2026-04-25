#!/usr/bin/env bash
# =============================================================
# multi_task_common.sh — 多任务混合训练共用配置
#
# 数据架构:
#   $MULTI_TASK_DATA_ROOT/
#   ├── base/          基座数据 (TG train + MCQ train)
#   ├── val/           固定 val (按任务分层采样)
#   └── experiments/   每个实验的混合训练数据
#
# 实验脚本用法:
#   source multi_task_common.sh
#   EXP_NAME="my_exp"
#   # (可选) REWARD_FUNCTION="..."
#   source run_multi_task.sh
# =============================================================

_COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${_COMMON_DIR}/.." && pwd)"

export DECORD_EOF_RETRY_MAX=2048001
export BAD_SAMPLES_LOG="${REPO_ROOT}/bad_samples.txt"

# ---- 实验标签 ----
PROJECT_NAME="${PROJECT_NAME:-EasyR1-multi-task}"

# ---- 模型 ----
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"

# ============================================================
# 数据路径 (统一管理)
# ============================================================
# 保持向后兼容: THREE_TASK_DATA_ROOT 仍可用
MULTI_TASK_DATA_ROOT="${MULTI_TASK_DATA_ROOT:-${THREE_TASK_DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task}}"
THREE_TASK_DATA_ROOT="${MULTI_TASK_DATA_ROOT}"  # backward compat

# -- Hier Seg 训练数据源 (20k, 每次按比例采样) --
HIER_TRAIN="${HIER_TRAIN:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/train/train_all.jsonl}"
HIER_VAL_SOURCE="${HIER_VAL_SOURCE:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/train/val_all.jsonl}"
HIER_TARGET="${HIER_TARGET:-2000}"

# -- Event Logic 训练数据源 --
EL_TRAIN="${EL_TRAIN:-}"
EL_VAL_SOURCE="${EL_VAL_SOURCE:-}"
EL_TARGET="${EL_TARGET:-2000}"
VAL_EL_N="${VAL_EL_N:-100}"

# -- 启用的任务列表 (空格分隔) --
TASKS="${TASKS:-tg mcq hier_seg}"

# -- 实验数据目录 (按 EXP_NAME 隔离) --
EXPERIMENTS_DIR="${MULTI_TASK_DATA_ROOT}/experiments"

# ============================================================
# 视频 & 序列配置
# ============================================================
DATA_IMAGE_DIR="${DATA_IMAGE_DIR:-}"
VIDEO_FPS="${VIDEO_FPS:-2.0}"
MAX_FRAMES="${MAX_FRAMES:-256}"
MAX_PIXELS="${MAX_PIXELS:-49152}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-14000}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-256}"

# ============================================================
# 硬件 (8卡默认, 可覆盖)
# ============================================================
ROLLOUT_BS="${ROLLOUT_BS:-32}"
GLOBAL_BS="${GLOBAL_BS:-32}"
MB_PER_UPDATE="${MB_PER_UPDATE:-1}"
MB_PER_EXP="${MB_PER_EXP:-2}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TP_SIZE="${TP_SIZE:-2}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.55}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-20480}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-true}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-true}"

# ============================================================
# 学习率
# ============================================================
LR="${LR:-5e-7}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
WARMUP_STYLE="${WARMUP_STYLE:-cosine}"

# ============================================================
# 核心算法
# ============================================================
ADV_ESTIMATOR="${ADV_ESTIMATOR:-ema_grpo}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
DISABLE_KL="${DISABLE_KL:-false}"
ONLINE_FILTERING="${ONLINE_FILTERING:-true}"
FILTER_LOW="${FILTER_LOW:-0.05}"
FILTER_HIGH="${FILTER_HIGH:-0.95}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.005}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"

# KL: 独立 loss, coef=0.04
KL_COEF="${KL_COEF:-0.04}"
KL_PENALTY="${KL_PENALTY:-low_var_kl}"

# ============================================================
# Reward (统一多任务: MCQ + TG + HierSeg F1-IoU)
# ============================================================
REWARD_FUNCTION="${REWARD_FUNCTION:-${REPO_ROOT}/verl/reward_function/mixed_proxy_reward.py:compute_score}"

# ============================================================
# 训练轮次 & 保存
# ============================================================
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-100}"
VAL_FREQ="${VAL_FREQ:-50}"
SAVE_LIMIT="${SAVE_LIMIT:-2}"
SAVE_BEST="${SAVE_BEST:-true}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task}"
FILLER_MODE="${FILLER_MODE:-signal}"
FILLER_START_DELAY="${FILLER_START_DELAY:-0}"
FILLER_PER_GPU="${FILLER_PER_GPU:-false}"
FILLER_BUSY_HOLD_MS="${FILLER_BUSY_HOLD_MS:-2200}"
FILLER_TARGET_UTIL="${FILLER_TARGET_UTIL:-88}"
FILLER_BUSY_MATRIX="${FILLER_BUSY_MATRIX:-3072}"
FILLER_BUSY_BATCH="${FILLER_BUSY_BATCH:-9}"
FILLER_BUSY_SLEEP_MS="${FILLER_BUSY_SLEEP_MS:-8}"
FILLER_IDLE_SLEEP_MS="${FILLER_IDLE_SLEEP_MS:-4}"
FILLER_ORPHAN_MATRIX="${FILLER_ORPHAN_MATRIX:-4096}"
FILLER_ORPHAN_BATCH="${FILLER_ORPHAN_BATCH:-22}"
FILLER_ORPHAN_SLEEP_MS="${FILLER_ORPHAN_SLEEP_MS:-6}"
FILLER_STALE_SIGNAL_TIMEOUT="${FILLER_STALE_SIGNAL_TIMEOUT:-120}"
