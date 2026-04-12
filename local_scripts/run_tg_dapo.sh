#!/usr/bin/env bash
# ============================================================
# Temporal Grounding 训练脚本 (2-GPU, EMA-GRPO)
#
# 使用 Time-R1 风格 <answer> 格式 + tIoU × distance_penalty reward.
# 单任务 (temporal_grounding) — 无需 task_homogeneous_batching.
#
# 用法:
#   bash local_scripts/run_tg_dapo.sh                    # 默认配置
#   MAX_STEPS=20 bash local_scripts/run_tg_dapo.sh       # 快速调试
#   N_GPUS_PER_NODE=8 bash local_scripts/run_tg_dapo.sh  # 8卡
# ============================================================
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# ---- 环境变量 ----
export DECORD_EOF_RETRY_MAX=2048001
export BAD_SAMPLES_LOG="$(pwd)/bad_samples.txt"
mkdir -p "$(dirname "${BAD_SAMPLES_LOG}")"

# ---- 实验配置 ----
project_name="${PROJECT_NAME:-EasyR1-temporal-grounding}"
exp_name="${EXP_NAME:-qwen3_vl_tg_dapo_2gpu}"

# ---- 模型 & 数据 ----
MODEL_PATH="${MODEL_PATH:-/home/xuboshen/models/Qwen3-VL-4B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/proxy_data/temporal_grounding/data/tg_train_max256s_cot.jsonl}"
TEST_FILE="${TEST_FILE:-${REPO_ROOT}/proxy_data/temporal_grounding/data/tg_val_max256s_cot.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-}"

# ---- 训练超参数 ----
ROLLOUT_BS="${ROLLOUT_BS:-8}"
GLOBAL_BS="${GLOBAL_BS:-8}"
MB_PER_UPDATE=1
MB_PER_EXP=1
ROLLOUT_N="${ROLLOUT_N:-8}"
TP_SIZE="${TP_SIZE:-2}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
NNODES="${NNODES:-1}"

# ---- 序列长度 & 视频 ----
MAX_PROMPT_LEN=14000
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-1024}"
VIDEO_FPS=2.0
MAX_FRAMES=256
MAX_PIXELS=49152
MIN_PIXELS=3136

# ---- 学习率 & 算法 ----
LR="${LR:-8e-7}"
ADV_ESTIMATOR=ema_grpo
ONLINE_FILTERING=false
DISABLE_KL=false
ENTROPY_COEFF=0.005
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.3

# ---- Reward: Temporal Grounding (tIoU × distance_penalty, [0, 2]) ----
REWARD_FUNCTION="${REWARD_FUNCTION:-${REPO_ROOT}/verl/reward_function/temporal_grounding_reward.py:compute_score}"

# ---- 训练步数 ----
MAX_STEPS="${MAX_STEPS:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
VAL_FREQ="${VAL_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-20}"

# ---- 启动训练 ----
TRAIN_CMD=(
    python3 -m verl.trainer.main
    config=examples/config_ema_grpo_64.yaml
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.image_dir="${IMAGE_DIR}"
    data.prompt_key="prompt"
    data.answer_key="answer"
    data.video_key="videos"
    data.video_fps="${VIDEO_FPS}"
    data.max_pixels="${MAX_PIXELS}"
    data.min_pixels="${MIN_PIXELS}"
    data.max_frames="${MAX_FRAMES}"
    data.max_prompt_length="${MAX_PROMPT_LEN}"
    data.max_response_length="${MAX_RESPONSE_LEN}"
    data.rollout_batch_size="${ROLLOUT_BS}"
    data.format_prompt=""
    data.filter_overlong_prompts=false
    data.task_homogeneous_batching=false
    algorithm.adv_estimator="${ADV_ESTIMATOR}"
    algorithm.disable_kl="${DISABLE_KL}"
    algorithm.use_kl_loss=true
    algorithm.online_filtering="${ONLINE_FILTERING}"
    worker.actor.global_batch_size="${GLOBAL_BS}"
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}"
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}"
    worker.actor.model.model_path="${MODEL_PATH}"
    worker.actor.fsdp.torch_dtype=bf16
    worker.actor.optim.strategy=adamw_bf16
    worker.actor.optim.lr="${LR}"
    worker.actor.clip_ratio_low="${CLIP_RATIO_LOW}"
    worker.actor.clip_ratio_high="${CLIP_RATIO_HIGH}"
    worker.actor.loss_avg_mode=token
    worker.actor.entropy_coeff="${ENTROPY_COEFF}"
    worker.rollout.n="${ROLLOUT_N}"
    worker.rollout.temperature=0.7
    worker.rollout.top_p=0.9
    worker.rollout.tensor_parallel_size="${TP_SIZE}"
    worker.rollout.gpu_memory_utilization=0.5
    worker.reward.reward_function="${REWARD_FUNCTION}"
    worker.reward.reward_type=batch
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.val_freq="${VAL_FREQ}"
    trainer.val_generations_to_log=4
    trainer.save_freq="${SAVE_FREQ}"
    trainer.logger="[file,tensorboard]"
    trainer.save_checkpoint_path="/m2v_intern/xuboshen/zgw/RL-Models/${exp_name}"
    data.val_batch_size=8
)

# 可选: 限制训练步数 (快速调试)
if [[ -n "${MAX_STEPS}" ]]; then
    TRAIN_CMD+=(trainer.max_steps="${MAX_STEPS}")
fi

"${TRAIN_CMD[@]}"
