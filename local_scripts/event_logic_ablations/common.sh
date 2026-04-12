#!/usr/bin/env bash
# =============================================================
# common.sh — Event Logic 消融实验共用超参数（重写版）
#
# 用法：在每个实验脚本里 source 本文件
# =============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export DECORD_EOF_RETRY_MAX=2048001
export BAD_SAMPLES_LOG="${REPO_ROOT}/bad_samples.txt"

# ---- 通用实验标签 ----
PROJECT_NAME="${PROJECT_NAME:-EasyR1-event-logic-ablation}"

# ---- 模型 ----
MODEL_PATH="${MODEL_PATH:-/home/xuboshen/models/Qwen3-VL-4B-Instruct}"

# ---- 数据源 ----
ANNOTATION_DIR="${ANNOTATION_DIR:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation/annotations_fixed_gmn25}"
CLIP_DIR="${CLIP_DIR:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation/clips}"

# ---- 消融实验数据根目录 ----
EL_DATA_ROOT="${EL_DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/event_logic}"

# ---- 视频 & 分辨率 ----
VIDEO_FPS=2.0
MAX_FRAMES=256
MAX_PIXELS=49152
MIN_PIXELS=3136

# ---- 序列长度 ----
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-14000}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"

# ---- 批次（8 卡默认）----
ROLLOUT_BS="${ROLLOUT_BS:-16}"
GLOBAL_BS="${GLOBAL_BS:-16}"
MB_PER_UPDATE="${MB_PER_UPDATE:-1}"
MB_PER_EXP="${MB_PER_EXP:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TP_SIZE="${TP_SIZE:-2}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}"

# ---- 学习率 ----
LR="${LR:-5e-7}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
WARMUP_STYLE="${WARMUP_STYLE:-cosine}"

# ---- 核心算法 ----
ADV_ESTIMATOR="${ADV_ESTIMATOR:-ema_grpo}"
DISABLE_KL=false
ONLINE_FILTERING="${ONLINE_FILTERING:-true}"
FILTER_LOW="${FILTER_LOW:-0.0}"
FILTER_HIGH="${FILTER_HIGH:-1.0}"
ENTROPY_COEFF=0.005
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.3

# ---- Reward: 使用 hier_seg reward（包含 sort） ----
REWARD_FUNCTION="${REWARD_FUNCTION:-${REPO_ROOT}/verl/reward_function/hier_seg_reward.py:compute_score}"

# ---- 训练轮次 & 保存 ----
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-60}"
SAVE_FREQ="${SAVE_FREQ:-20}"
VAL_FREQ="${VAL_FREQ:-10}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/event_logic/ablations}"
