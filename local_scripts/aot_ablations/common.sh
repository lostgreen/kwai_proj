#!/usr/bin/env bash
# =============================================================
# common.sh — Seg-AOT 消融实验共用超参数
#
# 用法：在每个实验脚本里 source 本文件，然后设置实验特有变量:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#   EXP_NAME=...
#   SEG_TASKS=...  # 空格分隔的任务列表
#   source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
# =============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export DECORD_EOF_RETRY_MAX=2048001
export BAD_SAMPLES_LOG="${REPO_ROOT}/bad_samples.txt"

# ---- 通用实验标签 ----
PROJECT_NAME="${PROJECT_NAME:-EasyR1-seg-aot-ablation}"
# EXP_NAME 由各实验脚本赋值

# ---- 模型 ----
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"

# ---- Seg Annotation 数据路径 ----
HIER_SEG_ROOT="${HIER_SEG_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation}"
ANNOTATION_DIR="${ANNOTATION_DIR:-${HIER_SEG_ROOT}/annotations_checked}"
CLIP_ROOT="${CLIP_ROOT:-${HIER_SEG_ROOT}/clips}"
# 原子 clips（prepare_all_clips.py 产出）: {CLIP_ROOT}/L1/, L2/, L3/
CLIP_DIR_L1="${CLIP_DIR_L1:-${CLIP_ROOT}/L1}"
CLIP_DIR_L2="${CLIP_DIR_L2:-${CLIP_ROOT}/L2}"
CLIP_DIR_L3="${CLIP_DIR_L3:-${CLIP_ROOT}/L3}"
SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-}"  # 空则从标注内 source_video_path 解析

# ---- 数据根目录 ----
SEG_AOT_DATA_ROOT="${SEG_AOT_DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_seg_aot}"

# ---- 视频 & 分辨率 ----
VIDEO_FPS=2.0
MAX_FRAMES=256
MAX_PIXELS=49152
MIN_PIXELS=3136

# ---- 序列长度 ----
MAX_PROMPT_LEN=14000
MAX_RESPONSE_LEN=1024

# ---- 批次（8 卡默认）----
ROLLOUT_BS="${ROLLOUT_BS:-16}"
GLOBAL_BS="${GLOBAL_BS:-16}"
MB_PER_UPDATE="${MB_PER_UPDATE:-1}"
MB_PER_EXP="${MB_PER_EXP:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.4}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"

# ---- 学习率（cosine 衰减）----
LR="${LR:-5e-7}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
WARMUP_STYLE="${WARMUP_STYLE:-cosine}"

# ---- 核心算法 ----
ADV_ESTIMATOR="${ADV_ESTIMATOR:-ema_grpo}"
DISABLE_KL=false
ONLINE_FILTERING="${ONLINE_FILTERING:-true}"
FILTER_LOW="${FILTER_LOW:-0.2}"
FILTER_HIGH="${FILTER_HIGH:-0.8}"
ENTROPY_COEFF=0.005
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.3

# ---- Reward ----
REWARD_FUNCTION="${REWARD_FUNCTION:-${REPO_ROOT}/verl/reward_function/mixed_proxy_reward.py:compute_score}"

# ---- 训练轮次 & 保存 ----
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-60}"
SAVE_FREQ="${SAVE_FREQ:-20}"
VAL_FREQ="${VAL_FREQ:-10}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/youcook2_seg_aot/ablations}"
FILLER_TARGET_UTIL="${FILLER_TARGET_UTIL:-80}"
