#!/usr/bin/env bash
# ============================================================
# run_multi_task.sh — 多任务混合训练入口
#
# 可直接运行或被消融实验脚本 source:
#   1) 直接运行: bash local_scripts/run_multi_task.sh
#   2) 消融实验: EXP_NAME=xxx source run_multi_task.sh
#
# 前提: 先运行 setup_base_data.sh 生成 base/ 和 val/
#
# 特点:
#   - freeze_vision_tower=true
#   - KL 独立 loss (use_kl_loss=true, kl_coef=0.04)
#   - EMA-GRPO + task homogeneous batching
#   - 1 epoch, val every 20 steps
# ============================================================
set -euo pipefail
set -x

# ---- Source common (如果还没 source 过) ----
if [[ -z "${REPO_ROOT:-}" ]]; then
    _SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    source "${_SCRIPT_DIR}/multi_task_common.sh"
fi

# ---- 实验配置 ----
EXP_NAME="${EXP_NAME:-multi_task_demo_2gpu}"
EXP_DATA_DIR="${EXPERIMENTS_DIR}/${EXP_NAME}"
TRAIN_FILE="${EXP_DATA_DIR}/train.jsonl"
TEST_FILE="${EXP_DATA_DIR}/val.jsonl"

# ============================================================
# Pre-flight: 检查 base/ val/ 是否存在
# ============================================================
# shellcheck disable=SC2086
python3 -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from local_scripts.data.mixer import main; main()
" \
    --data-root "${MULTI_TASK_DATA_ROOT}" \
    check \
    --tasks ${TASKS} \
    ${HIER_TRAIN:+--hier-train "${HIER_TRAIN}"} \
    ${EL_TRAIN:+--el-train "${EL_TRAIN}"} \
    ${EL_VAL_SOURCE:+--el-val-source "${EL_VAL_SOURCE}"} \
    --val-el-n "${VAL_EL_N}" \
|| { echo "[multi-task] Please run: bash local_scripts/setup_base_data.sh" >&2; exit 1; }

# ============================================================
# Step 0: 混合实验数据（仅首次）
# ============================================================
if [[ ! -f "${TRAIN_FILE}" ]] || [[ ! -f "${TEST_FILE}" ]]; then
    echo "[multi-task] Building experiment data for: ${EXP_NAME}"
    # shellcheck disable=SC2086
    python3 -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from local_scripts.data.mixer import main; main()
" \
        --data-root "${MULTI_TASK_DATA_ROOT}" \
        mix \
        --tasks ${TASKS} \
        --exp-name "${EXP_NAME}" \
        ${HIER_TRAIN:+--hier-train "${HIER_TRAIN}"} \
        ${HIER_TARGET:+--hier-target "${HIER_TARGET}"} \
        ${EL_TRAIN:+--el-train "${EL_TRAIN}"} \
        ${EL_TARGET:+--el-target "${EL_TARGET}"} \
        ${EL_VAL_SOURCE:+--el-val-source "${EL_VAL_SOURCE}"} \
        --val-el-n "${VAL_EL_N}"
    echo "[multi-task] Data ready: train=$(wc -l < "${TRAIN_FILE}"), val=$(wc -l < "${TEST_FILE}")"
fi

# ============================================================
# 任务采样权重（自动检测 problem_type）
# ============================================================
TASK_WEIGHT_MODE="${TASK_WEIGHT_MODE:-count}"
if [[ -z "${TASK_WEIGHTS:-}" ]]; then
    TASK_WEIGHTS="$(
python3 - "${TRAIN_FILE}" "${TASK_WEIGHT_MODE}" <<'PY'
import json, sys
from collections import Counter

path, mode = sys.argv[1], sys.argv[2].strip().lower()
counter = Counter()
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        task = str(json.loads(line).get("problem_type") or "").strip()
        if task:
            counter[task] += 1

if not counter:
    raise SystemExit(f"No problem_type found in {path}")

tasks = sorted(counter)
if mode in {"equal", "uniform"}:
    w = {t: 1.0 / len(tasks) for t in tasks}
elif mode in {"proportional", "count", "counts", "auto"}:
    total = sum(counter.values())
    w = {t: counter[t] / total for t in tasks}
else:
    raise SystemExit(f"Unsupported TASK_WEIGHT_MODE={mode!r}")

sys.stderr.write(f"Tasks: {', '.join(f'{t}={counter[t]}' for t in tasks)}\n")
sys.stderr.write(f"Weights ({mode}): {json.dumps(w)}\n")
print(json.dumps(w, separators=(",", ":")))
PY
)"
fi

# ============================================================
# GPU filler: 仅在训练结束或崩溃后启动（占满 GPU 防回收）
# 训练期间集群利用率已足够高，不需要 filler
# ============================================================
_filler_script="${REPO_ROOT}/local_scripts/gpu_filler.py"

_start_filler() {
  if [[ "${ENABLE_GPU_FILLER:-true}" != "true" ]] || [[ ! -f "${_filler_script}" ]]; then
    return
  fi
  # Kill any old instances first
  if pgrep -f "gpu_filler.py" > /dev/null 2>&1; then
    pkill -f "gpu_filler.py" 2>/dev/null || true
    sleep 2
  fi
  echo "[multi-task] Training stopped — starting GPU filler (target=${FILLER_TARGET_UTIL:-85}%)"
  nohup python3 "${_filler_script}" \
    --target-util "${FILLER_TARGET_UTIL:-85}" \
    --batch "${FILLER_BATCH:-50}" \
    --matrix-size "${FILLER_MATRIX:-8192}" \
    > /tmp/filler.log 2>&1 &
  echo "[multi-task] GPU filler started (PID $!), log: /tmp/filler.log"
}

# 训练结束（正常/崩溃）→ 清理信号文件 + 启动 filler
trap 'rm -f /tmp/verl_gpu_phase; _start_filler' EXIT

# ============================================================
# 启动训练
# ============================================================
python3 -m verl.trainer.main \
    config=examples/config_ema_grpo_64.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.image_dir="" \
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
    algorithm.kl_penalty="${KL_PENALTY}" \
    algorithm.kl_coef="${KL_COEF}" \
    algorithm.online_filtering="${ONLINE_FILTERING}" \
    algorithm.filter_low="${FILTER_LOW}" \
    algorithm.filter_high="${FILTER_HIGH}" \
    worker.actor.global_batch_size="${GLOBAL_BS}" \
    worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.freeze_vision_tower=true \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr="${LR}" \
    worker.actor.optim.lr_warmup_ratio="${LR_WARMUP_RATIO}" \
    worker.actor.optim.warmup_style="${WARMUP_STYLE}" \
    worker.actor.optim.min_lr_ratio="${LR_MIN_RATIO}" \
    worker.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    worker.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    worker.actor.loss_avg_mode=token \
    worker.actor.entropy_coeff="${ENTROPY_COEFF}" \
    worker.rollout.n="${ROLLOUT_N}" \
    worker.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    worker.rollout.top_p=0.9 \
    worker.rollout.tensor_parallel_size="${TP_SIZE}" \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.reward.reward_function="${REWARD_FUNCTION}" \
    worker.reward.reward_type=batch \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    ${MAX_STEPS:+trainer.max_steps="$MAX_STEPS"} \
    trainer.val_freq="${VAL_FREQ}" \
    trainer.val_generations_to_log=4 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.save_limit="${SAVE_LIMIT}" \
    trainer.save_best="${SAVE_BEST}" \
    trainer.logger="[file,tensorboard]" \
    trainer.save_checkpoint_path="${CHECKPOINT_ROOT}/${EXP_NAME}" \
    data.val_batch_size=8 \
    data.dataloader_num_workers="${DATALOADER_NUM_WORKERS:-2}"
