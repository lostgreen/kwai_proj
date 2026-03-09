#!/usr/bin/env bash
# ============================================================
# EasyR1 - Qwen3-VL YouCook2 视频时序分割 GRPO 训练脚本
# 配置：TP=2, 2卡, 全量微调（EasyR1 不支持 LoRA）
# ============================================================
set -x

# ---- 环境变量 ----
export DECORD_EOF_RETRY_MAX=2048001
# 将异常样本日志写到可写的绝对路径，避免 './EasyR1/bad_samples.txt' 不存在
export BAD_SAMPLES_LOG="$(pwd)/bad_samples.txt"
mkdir -p "$(dirname "${BAD_SAMPLES_LOG}")"
# 视频调试开关：打印采样帧数与模型侧视频输入信息
# export EASYR1_DEBUG_VIDEO_FRAMES="${EASYR1_DEBUG_VIDEO_FRAMES:-1}"
export EASYR1_DEBUG_VIDEO_FRAMES_MAX_LOGS="${EASYR1_DEBUG_VIDEO_FRAMES_MAX_LOGS:-200}"
# export WANDB_API_KEY=<YOUR_KEY>

# ---- 实验配置 ----
project_name='EasyR1-youcook2-segment'
exp_name='qwen3_vl_youcook2_temporal_seg_8gpu-48token'

# ---- 模型 & 数据 ----
MODEL_PATH="/home/xuboshen/models/Qwen3-VL-4B-Instruct"   # 替换为你的模型路径
TRAIN_FILE="/home/xuboshen/zgw/OneThinker/EasyR1/proxy_data/youcook2_train_easyr1.jsonl"     # 转换后的 EasyR1 格式数据
TEST_FILE="/home/xuboshen/zgw/EasyR1/proxy_data/youcook2_val_small.jsonl"      # 可替换为单独 val 集
IMAGE_DIR=""                                                # 视频已使用绝对路径则留空

# ---- 训练超参数（适配 2 卡）----
ROLLOUT_BS=16           # rollout batch size（2卡调小）
GLOBAL_BS=16            # actor 更新 global batch size
MB_PER_UPDATE=2         # 每设备每次更新 micro batch
MB_PER_EXP=2            # 每设备每次 experience 收集 micro batch
ROLLOUT_N=8             # 每个 prompt 生成的候选回复数（对应 ms-swift num_generations=8）
TP_SIZE=2               # vLLM Tensor Parallel size
N_GPUS_PER_NODE=8       # 每节点 GPU 数
NNODES=1                # 节点数

# ---- 序列长度 & 视频 ----
MAX_PROMPT_LEN=14000    # prompt 最大 token 数
MAX_RESPONSE_LEN=512    # 回复最大 token 数（thinking+events 内容 256 不够用）
VIDEO_FPS=2.0           # 视频采样帧率（effective 1fps after temporal_patch_size=2）
MAX_FRAMES=256          # 最大抽取帧数
MAX_PIXELS=49152      # 每帧最大像素数 qwen3 2帧合并一个patch 32 * 32 一个token
MIN_PIXELS=3136         # 每帧最小像素数（与 YAML 保持一致）

# ---- 学习率 & 算法 ----
LR=1e-6                 # 学习率
ADV_ESTIMATOR=grpo      # grpo

# ---- KL 正则化（关键！防止熵坍缩）----
DISABLE_KL=false        # 必须 false：启用 ref model 计算 KL loss
KL_COEF=0.2            # KL 系数适当加大，防止策略偏移过快

# ---- Reward ----
REWARD_FUNCTION="verl/reward_function/youcook2_temporal_seg_reward.py:compute_score"

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
    trainer.save_checkpoint_path="/m2v_intern/xuboshen/zgw/RL-Models/${exp_name}" \
    data.val_batch_size=8
