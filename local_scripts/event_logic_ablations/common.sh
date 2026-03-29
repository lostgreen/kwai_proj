#!/usr/bin/env bash
# =============================================================
# common.sh — Event Logic 消融实验共用超参数
#
# 用法：在每个实验脚本里 source 本文件，然后设置实验特有变量:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#   EXP_NAME=...
#   ADD_PER_VIDEO=...  REPLACE_PER_VIDEO=...  SORT_PER_VIDEO=...
# =============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export DECORD_EOF_RETRY_MAX=2048001
export BAD_SAMPLES_LOG="${REPO_ROOT}/bad_samples.txt"

# ---- 通用实验标签 ----
PROJECT_NAME="${PROJECT_NAME:-EasyR1-event-logic-ablation}"

# ---- 模型 ----
MODEL_PATH="${MODEL_PATH:-/home/xuboshen/models/Qwen3-VL-4B-Instruct}"

# ---- L2 标注数据源（管线 2，推荐）----
L2_ANNOTATION_DIR="${L2_ANNOTATION_DIR:-/m2v_intern/xuboshen/zgw/data/hier_seg_annotation/annotations}"
L2_CLIPS_DIR="${L2_CLIPS_DIR:-/m2v_intern/xuboshen/zgw/data/hier_seg_annotation/clips/L2}"
L2_FRAMES_DIR="${L2_FRAMES_DIR:-/m2v_intern/xuboshen/zgw/data/hier_seg_annotation/frames}"

# ---- 消融实验数据根目录 ----
EL_DATA_ROOT="${EL_DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/event_logic/ablations}"

# ---- AI 因果过滤参数 ----
FILTER_AI="${FILTER_AI:-false}"  # 是否启用 VLM 因果过滤
AI_API_BASE="${AI_API_BASE:-https://api.novita.ai/v3/openai}"
AI_MODEL="${AI_MODEL:-qwen/qwen2.5-vl-72b-instruct}"
AI_CONFIDENCE="${AI_CONFIDENCE:-0.75}"
AI_FILTER_WORKERS="${AI_FILTER_WORKERS:-4}"

# ---- build_l2_event_logic.py 共用参数 ----
MIN_EVENTS="${MIN_EVENTS:-4}"
MIN_CONTEXT="${MIN_CONTEXT:-2}"
MAX_CONTEXT="${MAX_CONTEXT:-4}"
REPLACE_SEQ_LEN="${REPLACE_SEQ_LEN:-5}"
SORT_SEQ_LEN="${SORT_SEQ_LEN:-5}"
BUILD_SEED="${BUILD_SEED:-42}"

# ---- 验证集 ----
TEST_FILE="${TEST_FILE:-}"

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
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"

# ---- 离线过滤 ----
FILTER_ROLLOUT_N="${FILTER_ROLLOUT_N:-16}"
FILTER_NUM_GPUS="${FILTER_NUM_GPUS:-${N_GPUS_PER_NODE}}"
FILTER_GPU_MEM_UTIL="${FILTER_GPU_MEM_UTIL:-0.7}"
FILTER_MAX_MODEL_LEN="${FILTER_MAX_MODEL_LEN:-16384}"

# ---- 学习率 ----
LR="${LR:-5e-7}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
WARMUP_STYLE="${WARMUP_STYLE:-cosine}"

# ---- 核心算法 ----
ADV_ESTIMATOR="${ADV_ESTIMATOR:-ema_grpo}"
DISABLE_KL=false
ONLINE_FILTERING="${ONLINE_FILTERING:-true}"
TASK_HOMOGENEOUS="${TASK_HOMOGENEOUS:-true}"
ENTROPY_COEFF=0.005
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.3

# ---- Reward ----
REWARD_FUNCTION="${REWARD_FUNCTION:-${REPO_ROOT}/verl/reward_function/mixed_proxy_reward.py:compute_score}"

# ---- 训练轮次 & 保存 ----
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-20}"
VAL_FREQ="${VAL_FREQ:-10}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/event_logic/ablations}"

# ---- 难度采样 ----
SKIP_CURATE="${SKIP_CURATE:-true}"

# ---- 任务权重 ----
TASK_WEIGHT_MODE="${TASK_WEIGHT_MODE:-count}"
