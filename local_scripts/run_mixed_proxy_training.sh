#!/usr/bin/env bash
# ============================================================
# 混合训练脚本: 代理任务 (add/delete/replace/sort) + 时序分割 (temporal_seg)
#
# 核心特性:
#   1) task_homogeneous_batching=true: 每个 batch 只包含同一任务
#   2) task_weights: 控制各任务采样比例
#   3) mixed_proxy_reward: 统一 reward 函数自动按 problem_type 分派
#   4) ema_grpo: 按 problem_type 做 EMA 标准差归一化
# ============================================================
set -x

# ---- 环境变量 ----
export DECORD_EOF_RETRY_MAX=2048001
export BAD_SAMPLES_LOG="$(pwd)/bad_samples.txt"
mkdir -p "$(dirname "${BAD_SAMPLES_LOG}")"

# ---- 实验配置 ----
project_name='EasyR1-mixed-proxy'
exp_name='qwen3_vl_mixed_proxy_training'

# ---- 模型 & 数据 ----
MODEL_PATH="/home/xuboshen/models/Qwen3-VL-4B-Instruct"   # 替换为你的模型路径
TRAIN_FILE="proxy_data/mixed_train_cot.jsonl"              # CoT prompt 版本（选择题要求 <think>...<answer>）
TEST_FILE="proxy_data/youcook2_val_small.jsonl"             # 验证集
IMAGE_DIR=""                                                 # 视频已使用绝对路径则留空

# ---- 训练超参数 ----
ROLLOUT_BS=16           # rollout batch size
GLOBAL_BS=16            # actor 更新 global batch size
MB_PER_UPDATE=1         # 每设备每次更新 micro batch
MB_PER_EXP=1            # 每设备每次 experience 收集 micro batch
ROLLOUT_N=8             # 每个 prompt 生成的候选回复数
TP_SIZE=2               # vLLM Tensor Parallel size
N_GPUS_PER_NODE=8       # 每节点 GPU 数
NNODES=1                # 节点数

# ---- 序列长度 & 视频 ----
MAX_PROMPT_LEN=14000    # prompt 最大 token 数
MAX_RESPONSE_LEN=1024   # CoT 推理需要更长回复（旧值 512 对选择题太短）
VIDEO_FPS=2.0
MAX_FRAMES=256
MAX_PIXELS=49152
MIN_PIXELS=3136

# ---- 学习率 & 算法 ----
LR=8e-7                 # 混合训练建议比单任务低 20~30%
ADV_ESTIMATOR=ema_grpo  # 使用 EMA-GRPO, 按 problem_type 分任务归一化

# ---- KL 正则化 ----
DISABLE_KL=false
KL_COEF=0.1

# ---- 任务采样权重 ----
# temporal_seg 占 40%, 4 种代理任务各占 15%
TASK_WEIGHTS='{"temporal_seg":0.40,"add":0.15,"delete":0.15,"replace":0.15,"sort":0.15}'

# ---- Reward (统一多任务 reward 函数) ----
REWARD_FUNCTION="verl/reward_function/mixed_proxy_reward.py:compute_score"

# ---- 启动训练 ----
python3 -m verl.trainer.main \
    config=examples/config_ema_grpo_64.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.image_dir="${IMAGE_DIR}" \
    data.prompt_key="prompt" \
    data.answer_key="answer" \
    data.video_key="videos" \
    data.video_fps="${VIDEO_FPS}" \
    data.max_pixels="${MAX_PIXELS}" \
    data.min_pixels="${MIN_PIXELS}" \
    data.max_frames="${MAX_FRAMES}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    data.rollout_batch_size="${ROLLOUT_BS}" \
    data.format_prompt="" \
    data.filter_overlong_prompts=false \
    data.task_homogeneous_batching=true \
    data.task_weights="${TASK_WEIGHTS}" \
    data.task_key="problem_type" \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.disable_kl="${DISABLE_KL}" \
    algorithm.use_kl_loss=true \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_coef="${KL_COEF}" \
    algorithm.online_filtering=false \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr="${LR}" \
    worker.rollout.n="${ROLLOUT_N}" \
    worker.rollout.temperature=0.7 \
    worker.rollout.top_p=0.9 \
    worker.rollout.tensor_parallel_size="${TP_SIZE}" \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.reward.reward_function="${REWARD_FUNCTION}" \
    worker.reward.reward_type=batch \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.total_epochs=1 \
    trainer.val_freq=100 \
    trainer.val_generations_to_log=4 \
    trainer.save_freq=50 \
    trainer.logger="[file,tensorboard]" \
    trainer.save_checkpoint_path="checkpoints/${exp_name}" \
    data.val_batch_size=8
