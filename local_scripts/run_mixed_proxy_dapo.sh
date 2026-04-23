#!/usr/bin/env bash
# ============================================================
# 混合训练脚本:
#   EMA-GRPO + KL loss + 离线 rollout 筛选
#
# 与 run_mixed_proxy_training.sh 的核心区别:
#   1) 训练前先离线 rollout 一遍训练集，仅保留 8 个 rollout 奖励不全相同的样本
#   2) 训练时关闭 online filtering，保留基础框架行为
#   3) adv_estimator=ema_grpo  — 保留 EMA 基线
#   4) use_kl_loss=true        — EMA-GRPO 使用 ref model + KL loss
#
# 详细机制请参见: docs/ema_dapo_mechanism.md
# ============================================================
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/local_scripts/gpu_filler_common.sh"

# ---- 环境变量 ----
export DECORD_EOF_RETRY_MAX=2048001
export BAD_SAMPLES_LOG="$(pwd)/bad_samples.txt"
mkdir -p "$(dirname "${BAD_SAMPLES_LOG}")"

# ---- 实验配置 ----
project_name="${PROJECT_NAME:-EasyR1-temporal-aot}"
exp_name="${EXP_NAME:-qwen3_vl_temporal_aot_dapo_2gpu_offline_filtered}"

# ---- 模型 & 数据 ----
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"   # 替换为你的模型路径
TRAIN_SOURCE_FILE="${TRAIN_FILE:-${REPO_ROOT}/proxy_data/temporal_aot/data/mixed_aot_train.jsonl}"
TEST_FILE="${TEST_FILE:-${REPO_ROOT}/proxy_data/temporal_aot/data/mixed_aot_val.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-}"                                                 # 视频已使用绝对路径则留空
TRAIN_FILE="${TRAIN_SOURCE_FILE}"

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

# ---- 核心算法参数 ----
ADV_ESTIMATOR=ema_grpo

# 训练阶段关闭 online filtering，改为训练前离线筛选。
ONLINE_FILTERING=false

# EMA-GRPO 需要 ref model + KL loss
DISABLE_KL=false

# Entropy 正则
ENTROPY_COEFF=0.005

# Clip-Higher: DAPO 非对称 clip（允许更大的正向更新）
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.3

# ---- Reward (统一多任务 reward 函数, 严格格式模式) ----
REWARD_FUNCTION="${REWARD_FUNCTION:-${REPO_ROOT}/verl/reward_function/mixed_proxy_reward.py:compute_score}"

# ---- 离线 rollout 筛选 ----
OFFLINE_FILTER="${OFFLINE_FILTER:-true}"
OFFLINE_FILTER_FORCE="${OFFLINE_FILTER_FORCE:-false}"
OFFLINE_FILTER_MAX_SAMPLES="${OFFLINE_FILTER_MAX_SAMPLES:-0}"
OFFLINE_FILTER_OUTPUT="${OFFLINE_FILTER_OUTPUT:-${TRAIN_SOURCE_FILE%.jsonl}.offline_filtered.jsonl}"
OFFLINE_FILTER_REPORT="${OFFLINE_FILTER_REPORT:-${TRAIN_SOURCE_FILE%.jsonl}.offline_filter_report.jsonl}"
OFFLINE_FILTER_BACKEND="${OFFLINE_FILTER_BACKEND:-vllm}"
OFFLINE_FILTER_GPU_MEMORY_UTILIZATION="${OFFLINE_FILTER_GPU_MEMORY_UTILIZATION:-0.8}"
OFFLINE_FILTER_MAX_MODEL_LEN="${OFFLINE_FILTER_MAX_MODEL_LEN:-0}"
OFFLINE_FILTER_MAX_BATCHED_TOKENS="${OFFLINE_FILTER_MAX_BATCHED_TOKENS:-16384}"

if [[ "${OFFLINE_FILTER}" == "true" ]]; then
    if [[ "${OFFLINE_FILTER_FORCE}" == "true" || ! -f "${OFFLINE_FILTER_OUTPUT}" ]]; then
        python3 "${REPO_ROOT}/local_scripts/offline_rollout_filter.py" \
            --input_jsonl "${TRAIN_SOURCE_FILE}" \
            --output_jsonl "${OFFLINE_FILTER_OUTPUT}" \
            --report_jsonl "${OFFLINE_FILTER_REPORT}" \
            --model_path "${MODEL_PATH}" \
            --reward_function "${REWARD_FUNCTION}" \
            --backend "${OFFLINE_FILTER_BACKEND}" \
            --num_rollouts "${ROLLOUT_N}" \
            --temperature 0.7 \
            --top_p 0.9 \
            --max_new_tokens "${MAX_RESPONSE_LEN}" \
            --video_fps "${VIDEO_FPS}" \
            --max_frames "${MAX_FRAMES}" \
            --max_pixels "${MAX_PIXELS}" \
            --min_pixels "${MIN_PIXELS}" \
            --max_samples "${OFFLINE_FILTER_MAX_SAMPLES}" \
            --tensor_parallel_size "${TP_SIZE}" \
            --gpu_memory_utilization "${OFFLINE_FILTER_GPU_MEMORY_UTILIZATION}" \
            --max_model_len "${OFFLINE_FILTER_MAX_MODEL_LEN}" \
            --max_num_batched_tokens "${OFFLINE_FILTER_MAX_BATCHED_TOKENS}"
    fi
    TRAIN_FILE="${OFFLINE_FILTER_OUTPUT}"
fi

# ---- 任务采样权重 ----
# 自动从最终训练集读取 problem_type，避免每次换数据手改权重。
# TASK_WEIGHT_MODE:
#   - equal / uniform: 对训练集中出现的每个任务平均采样
#   - proportional / count: 按训练集样本占比采样
TASK_WEIGHT_MODE="${TASK_WEIGHT_MODE:-count}"
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

# ---- 启动训练 ----
trap 'gpu_filler_clear_signal' EXIT
gpu_filler_start "[mixed-proxy]"

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
    data.task_homogeneous_batching=false \
    data.task_weights="${TASK_WEIGHTS}" \
    data.task_key="problem_type" \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.disable_kl="${DISABLE_KL}" \
    algorithm.use_kl_loss=true \
    algorithm.online_filtering="${ONLINE_FILTERING}" \
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
    trainer.val_freq=10 \
    trainer.val_generations_to_log=4 \
    trainer.save_freq=20 \
    trainer.logger="[file,tensorboard]" \
    trainer.save_checkpoint_path="/m2v_intern/xuboshen/zgw/RL-Models/${exp_name}" \
    data.val_batch_size=8

gpu_filler_clear_signal
