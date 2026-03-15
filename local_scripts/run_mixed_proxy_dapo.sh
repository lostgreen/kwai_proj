#!/usr/bin/env bash
# ============================================================
# 混合训练脚本 (DAPO 变体):
#   EMA-GRPO advantage + DAPO 动态过滤 + Clip-Higher + Token-Loss + Entropy 正则
#
# 与 run_mixed_proxy_training.sh 的核心区别:
#   1) online_filtering=true  — 动态剔除"全对组"和"全错组"，仅保留有学习信号的组
#   2) disable_kl=true         — 移除 KL 惩罚项（DAPO 原则：不限制探索）
#   3) entropy_coeff=0.005     — Entropy 正则，防止动态过滤后模式坍塌
#   4) clip_ratio_high=0.3     — Clip-Higher，允许更大的正向更新步长
#   5) loss_avg_mode=token     — Token-level 损失（DAPO 标准做法）
#   6) adv_estimator=ema_grpo  — 保留 EMA 基线（多任务奖励尺度归一化）
#
# 详细机制请参见: docs/ema_dapo_mechanism.md
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
project_name="${PROJECT_NAME:-EasyR1-temporal-aot}"
exp_name="${EXP_NAME:-qwen3_vl_temporal_aot_dapo_2gpu_filtered}"

# ---- 模型 & 数据 ----
MODEL_PATH="${MODEL_PATH:-/home/xuboshen/models/Qwen3-VL-4B-Instruct}"   # 替换为你的模型路径
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/proxy_data/temporal_aot/data/mixed_aot_train.jsonl}"
TEST_FILE="${TEST_FILE:-${REPO_ROOT}/proxy_data/temporal_aot/data/mixed_aot_val.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-}"                                                 # 视频已使用绝对路径则留空

# ---- 训练超参数 ----
ROLLOUT_BS=8           # rollout batch size
GLOBAL_BS=8           # actor 更新 global batch size
MB_PER_UPDATE=1         # 每设备每次更新 micro batch
MB_PER_EXP=1            # 每设备每次 experience 收集 micro batch
ROLLOUT_N=8             # 每个 prompt 生成的候选回复数
TP_SIZE=2               # vLLM Tensor Parallel size
N_GPUS_PER_NODE=2       # 每节点 GPU 数
NNODES=1                # 节点数

# ---- 序列长度 & 视频 ----
MAX_PROMPT_LEN=14000    # prompt 最大 token 数
MAX_RESPONSE_LEN=1024   # CoT 推理需要更长回复
VIDEO_FPS=2.0
MAX_FRAMES=256
MAX_PIXELS=49152
MIN_PIXELS=3136

# ---- 学习率 & 算法 ----
LR=8e-7

# ---- DAPO 核心参数 ----
# adv_estimator: 保留 EMA-GRPO（多任务奖励尺度归一化的关键）
ADV_ESTIMATOR=ema_grpo

# online_filtering: DAPO 动态过滤
# 过滤单位是“同一个 prompt 的 ROLLOUT_N 个回复”。
# 仅当组均值满足 filter_low < mean(reward[filter_key]) < filter_high 时才参与更新。
# 因此均值为 0.0 / 1.0 的退化组会被剔除；若连续多次都没有保留样本，则回退到未过滤 batch。
ONLINE_FILTERING=true
FILTER_LOW=0.01
FILTER_HIGH=0.99
MAX_TRY_MAKE_BATCH=10

# DAPO: 关闭 KL 惩罚（允许模型大步探索新格式，不被 ref policy 约束）
DISABLE_KL=true

# Entropy 正则: 防止动态过滤后 rollout 分布过度收窄
# 推荐范围: 0.001 ~ 0.01; 0.0 = 关闭
ENTROPY_COEFF=0.005

# Clip-Higher: DAPO 非对称 clip（允许更大的正向更新）
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.3

# ---- 任务采样权重 ----
# 自动从训练集读取 problem_type，避免每次换数据手改权重。
# TASK_WEIGHT_MODE:
#   - equal / uniform: 对训练集中出现的每个任务平均采样
#   - proportional / count: 按训练集样本占比采样
TASK_WEIGHT_MODE="${TASK_WEIGHT_MODE:-equal}"
if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "Train file not found: ${TRAIN_FILE}" >&2
    exit 1
fi
if [[ -z "${TASK_WEIGHTS:-}" ]]; then
    TASK_WEIGHTS="$(
python3 - "${TRAIN_FILE}" "${TASK_WEIGHT_MODE}" <<'PY'
import json
import sys
from collections import Counter

path = sys.argv[1]
mode = sys.argv[2].strip().lower()
counter = Counter()

with open(path, encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse {path}:{line_no}: {exc}")
        task = str(obj.get("problem_type") or "").strip()
        if task:
            counter[task] += 1

if not counter:
    raise SystemExit(f"No non-empty problem_type found in {path}")

task_names = sorted(counter)
if mode in {"equal", "uniform"}:
    weights = {task: 1.0 / len(task_names) for task in task_names}
elif mode in {"proportional", "count", "counts", "auto"}:
    total = sum(counter.values())
    weights = {task: counter[task] / total for task in task_names}
else:
    raise SystemExit(
        f"Unsupported TASK_WEIGHT_MODE={mode!r}. "
        "Use equal/uniform or proportional/count."
    )

summary = ", ".join(f"{task}={counter[task]}" for task in task_names)
sys.stderr.write(f"Detected task counts from {path}: {summary}\n")
sys.stderr.write(f"Using task weights ({mode}): {json.dumps(weights, ensure_ascii=False)}\n")
print(json.dumps(weights, ensure_ascii=False, separators=(",", ":")))
PY
)"
fi

# ---- Reward (统一多任务 reward 函数, 严格格式模式) ----
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
    algorithm.use_kl_loss=false \
    algorithm.online_filtering="${ONLINE_FILTERING}" \
    algorithm.filter_key=overall \
    algorithm.filter_low="${FILTER_LOW}" \
    algorithm.filter_high="${FILTER_HIGH}" \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr="${LR}" \
    worker.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    worker.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    worker.actor.loss_avg_mode=token \
    worker.actor.entropy_coeff="${ENTROPY_COEFF}" \
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
    trainer.max_try_make_batch="${MAX_TRY_MAKE_BATCH}" \
    trainer.val_freq=10 \
    trainer.val_generations_to_log=4 \
    trainer.save_freq=20 \
    trainer.save_filtered_rollout_to_file=true \
    trainer.save_filtered_rollout_n_per_step=-1 \
    trainer.logger="[file,tensorboard]" \
    trainer.save_checkpoint_path="/m2v_intern/xuboshen/zgw/RL-Models/${exp_name}" \
    data.val_batch_size=8
