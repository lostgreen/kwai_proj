#!/usr/bin/env bash
# =============================================================
# common.sh — AoT 消融实验共用超参数
#
# 用法：在每个实验脚本里 source 本文件，然后设置实验特有变量:
#   source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#   EXP_NAME=...
#   DATA_DIR=...
#   # 定义 MCQ 构造参数（见各 expN.sh）
# =============================================================

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export DECORD_EOF_RETRY_MAX=2048001
export BAD_SAMPLES_LOG="${REPO_ROOT}/bad_samples.txt"

# ---- 通用实验标签 ----
PROJECT_NAME="${PROJECT_NAME:-EasyR1-aot-ablation}"
# EXP_NAME 由各实验脚本赋值

# ---- 模型 ----
MODEL_PATH="${MODEL_PATH:-//home/xuboshen/models/Qwen3-VL-4B-Instruct}"

# ---- 上游标注数据（Step 1-2 产出，所有实验共享）----
AOT_DATA_ROOT="${AOT_DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot}"
MANIFEST_JSONL="${MANIFEST_JSONL:-${AOT_DATA_ROOT}/aot_event_manifest.jsonl}"
CAPTION_PAIRS="${CAPTION_PAIRS:-${AOT_DATA_ROOT}/caption_pairs.jsonl}"
# 消融实验默认不混合 temporal_seg，纯 AoT MCQ 训练；如需混合可覆盖
SEG_JSONL="${SEG_JSONL:-}"

# ---- 验证集（所有实验共用）----
# 所有实验共用验证集（训练 Step E 的 val_files），需事先构建
TEST_FILE="${TEST_FILE:-${REPO_ROOT}/proxy_data/temporal_aot/data/aot_ablation_val.jsonl}"

# ---- MCQ 构造参数（各实验可覆盖）----
MCQ_MAX_SAMPLES="${MCQ_MAX_SAMPLES:-2000}"   # aot proxy 样本总量（实验间对齐）
MCQ_MIN_CONFIDENCE="${MCQ_MIN_CONFIDENCE:-0.6}"

# ---- 视频 & 分辨率 ----
VIDEO_FPS=2.0
MAX_FRAMES=256
MAX_PIXELS=49152
MIN_PIXELS=3136

# ---- 序列长度 ----
MAX_PROMPT_LEN=14000
MAX_RESPONSE_LEN=1024

# ---- 批次（8 卡默认）----
ROLLOUT_BS="${ROLLOUT_BS:-32}"
GLOBAL_BS="${GLOBAL_BS:-32}"
MB_PER_UPDATE="${MB_PER_UPDATE:-1}"
MB_PER_EXP="${MB_PER_EXP:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
TP_SIZE="${TP_SIZE:-2}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"

# ---- 学习率（cosine 衰减） ----
LR="${LR:-2e-6}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.05}"   # 前 5% steps warmup
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"           # 最终 LR = LR * 0.1
WARMUP_STYLE="${WARMUP_STYLE:-cosine}"        # cosine | constant

# ---- 核心算法 ----
ADV_ESTIMATOR=ema_grpo
DISABLE_KL=false
ONLINE_FILTERING=false
ENTROPY_COEFF=0.005
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.3

# ---- Reward ----
REWARD_FUNCTION="${REWARD_FUNCTION:-${REPO_ROOT}/verl/reward_function/mixed_proxy_reward.py:compute_score}"

# ---- 训练轮次 & 保存 ----
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-20}"
VAL_FREQ="${VAL_FREQ:-10}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/youcook2_aot/ablations}"

# ---- 任务权重模式 ----
TASK_WEIGHT_MODE="${TASK_WEIGHT_MODE:-count}"
