#!/usr/bin/env bash
# =============================================================
# launch_seg_train.sh — Seg-AOT 消融实验统一训练入口（纯训练）
#
# 前置条件: 数据已通过 proxy_data/.../build_data.sh 构造完毕
#
# 调用前必须在各实验脚本里已 source common.sh 并设置:
#   EXP_NAME       实验名称
#   SEG_TASKS      空格分隔的任务列表
# =============================================================
set -euo pipefail
set -x

if [[ -z "${EXP_NAME:-}" ]]; then echo "[seg-aot] EXP_NAME not set" >&2; exit 1; fi
if [[ -z "${SEG_TASKS:-}" ]]; then echo "[seg-aot] SEG_TASKS not set" >&2; exit 1; fi
source "${REPO_ROOT}/local_scripts/gpu_filler_common.sh"

DATA_DIR="${DATA_DIR:-${SEG_AOT_DATA_ROOT}/${DATA_NAME:-${EXP_NAME}}}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.jsonl}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/val.jsonl}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[seg-aot] ERROR: TRAIN_FILE not found: ${TRAIN_FILE}" >&2
  echo "[seg-aot] Please run build_data.sh first:" >&2
  echo "[seg-aot]   bash proxy_data/youcook2_seg/temporal_aot/build_data.sh" >&2
  exit 1
fi

# ---- 运行日志 ----
_ckpt_dir="${CHECKPOINT_ROOT}/${EXP_NAME}"
mkdir -p "${_ckpt_dir}"
_ts="$(date +%Y%m%d_%H%M%S)"
_run_log="${_ckpt_dir}/run_${_ts}.log"
exec > >(tee -a "${_run_log}") 2>&1
echo "[seg-aot] ============================================================"
echo "[seg-aot] EXP    : ${EXP_NAME}"
echo "[seg-aot] Tasks  : ${SEG_TASKS}"
echo "[seg-aot] Date   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "[seg-aot] Train  : ${TRAIN_FILE}  ($(wc -l < "${TRAIN_FILE}") samples)"
echo "[seg-aot] Val    : ${TEST_FILE}   ($(wc -l < "${TEST_FILE}") samples)"
echo "[seg-aot] ============================================================"

# Ray tmpdir
_exp_short="${EXP_NAME:0:12}"
_ray_tmpdir="/tmp/ray_${_exp_short}"
mkdir -p "${_ray_tmpdir}"
export RAY_TMPDIR="${_ray_tmpdir}"

# ---- GPU filler: 训练期间常驻，训练退出后仅清理 signal ----
trap 'gpu_filler_clear_signal' EXIT
gpu_filler_start "[seg-aot]"

# ---- Step B: 训练 ----
echo "[seg-aot] Starting training: ${EXP_NAME}"

TENSORBOARD_DIR="${CHECKPOINT_ROOT}/${EXP_NAME}/tensorboard" \
python3 -m verl.trainer.main \
  config="${REPO_ROOT}/examples/config_ema_grpo_64.yaml" \
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
  data.task_key="problem_type" \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  algorithm.disable_kl="${DISABLE_KL}" \
  algorithm.use_kl_loss=true \
  algorithm.online_filtering="${ONLINE_FILTERING}" \
  algorithm.filter_low="${FILTER_LOW}" \
  algorithm.filter_high="${FILTER_HIGH}" \
  worker.actor.global_batch_size="${GLOBAL_BS}" \
  worker.actor.micro_batch_size_per_device_for_update="${MB_PER_UPDATE}" \
  worker.actor.micro_batch_size_per_device_for_experience="${MB_PER_EXP}" \
  worker.actor.model.model_path="${MODEL_PATH}" \
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
  worker.rollout.temperature=0.7 \
  worker.rollout.top_p=0.9 \
  worker.rollout.tensor_parallel_size="${TP_SIZE}" \
  worker.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  worker.reward.reward_function="${REWARD_FUNCTION}" \
  worker.reward.reward_type=batch \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_freq="${VAL_FREQ}" \
  trainer.val_generations_to_log=4 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.logger="[file,tensorboard]" \
  trainer.save_checkpoint_path="${CHECKPOINT_ROOT}/${EXP_NAME}" \
  data.val_batch_size=8

gpu_filler_clear_signal

echo "[seg-aot] Done. Log: ${_run_log}"
