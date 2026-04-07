#!/usr/bin/env bash
# =============================================================
# launch_train.sh — Grounding+Seg 消融实验统一训练入口
#
# 调用前必须在各实验脚本里已设置（并 source common.sh）：
#   EXP_NAME      实验名称
#   TRAIN_FILE    训练 JSONL 路径
#   TEST_FILE     验证 JSONL 路径
# =============================================================
set -euo pipefail

# ---- 前置检查 ----
if [[ -z "${EXP_NAME:-}" ]];   then echo "[gseg] EXP_NAME not set"   >&2; exit 1; fi
if [[ -z "${TRAIN_FILE:-}" ]]; then echo "[gseg] TRAIN_FILE not set" >&2; exit 1; fi
if [[ -z "${TEST_FILE:-}" ]];  then echo "[gseg] TEST_FILE not set"  >&2; exit 1; fi

if [[ ! -f "${TRAIN_FILE}" ]]; then echo "[gseg] TRAIN_FILE not found: ${TRAIN_FILE}" >&2; exit 1; fi
if [[ ! -f "${TEST_FILE}" ]];  then echo "[gseg] TEST_FILE not found: ${TEST_FILE}"  >&2; exit 1; fi

# =========================================================
# 运行日志 & Ray 会话目录
# =========================================================
_ckpt_dir="${CHECKPOINT_ROOT}/${EXP_NAME}"
mkdir -p "${_ckpt_dir}"
_ts="$(date +%Y%m%d_%H%M%S)"
_run_log="${_ckpt_dir}/run_${_ts}.log"
_summary_log="${_ckpt_dir}/summary_${_ts}.log"
exec > >(tee >(grep --line-buffered -E '^\[gseg\]|ERROR|Error|Exception|Traceback|SIGTERM|SIGKILL|OOM|killed|BAD_SAMPLE|step [0-9]|reward|val.*metric' >> "${_summary_log}") -a "${_run_log}") 2>&1
echo "[gseg] ============================================================"
echo "[gseg] EXP    : ${EXP_NAME}"
echo "[gseg] Date   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "[gseg] Log    : ${_run_log}"
echo "[gseg] Summary: ${_summary_log}"
echo "[gseg] Train  : ${TRAIN_FILE}  ($(wc -l < "${TRAIN_FILE}") samples)"
echo "[gseg] Val    : ${TEST_FILE}   ($(wc -l < "${TEST_FILE}") samples)"
echo "[gseg] GPUs   : ${N_GPUS_PER_NODE}  TP=${TP_SIZE}"
echo "[gseg] ============================================================"

# Ray tmpdir: short name to avoid AF_UNIX socket path >107 bytes
_exp_short="$(echo "${EXP_NAME}" | grep -oE 'exp[0-9]+' || echo "${EXP_NAME:0:8}")"
_ray_tmpdir="/tmp/ray_${_exp_short}"
mkdir -p "${_ray_tmpdir}"
export RAY_TMPDIR="${_ray_tmpdir}"
echo "[gseg] Ray tmpdir: ${_ray_tmpdir}"

# =========================================================
# GPU filler: 保持利用率 ≥ 80%
# =========================================================
_filler_script="${REPO_ROOT}/local_scripts/gpu_filler.py"
if [[ "${ENABLE_GPU_FILLER:-true}" == "true" ]] && [[ -f "${_filler_script}" ]]; then
  if pgrep -f "gpu_filler.py" > /dev/null 2>&1; then
    echo "[gseg] Killing old filler instances..."
    pkill -f "gpu_filler.py" 2>/dev/null || true
    sleep 2
  fi
  echo "[gseg] Starting GPU filler (target=${FILLER_TARGET_UTIL:-85}%)"
  nohup python3 "${_filler_script}" \
    --target-util "${FILLER_TARGET_UTIL:-85}" \
    --batch "${FILLER_BATCH:-50}" \
    --matrix-size "${FILLER_MATRIX:-8192}" \
    > /tmp/filler.log 2>&1 &
  echo "[gseg] GPU filler started (PID $!)"
fi

trap 'rm -f /tmp/verl_gpu_phase' EXIT

# =========================================================
# 启动训练
# =========================================================
echo "[gseg] Starting training: ${EXP_NAME}"
echo "[gseg] LR=${LR}  warmup=${WARMUP_STYLE}  warmup_ratio=${LR_WARMUP_RATIO}"

set -x
TENSORBOARD_DIR="${CHECKPOINT_ROOT}/${EXP_NAME}/tensorboard" \
TENSORBOARD_FLAT=1 \
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
  worker.rollout.gpu_memory_utilization=0.35 \
  worker.reward.reward_function="${REWARD_FUNCTION}" \
  worker.reward.reward_type=batch \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.val_freq="${VAL_FREQ}" \
  trainer.val_generations_to_log=4 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.logger="[file,tensorboard]" \
  trainer.save_checkpoint_path="${CHECKPOINT_ROOT}/${EXP_NAME}" \
  data.val_batch_size=8 \
  ${MAX_STEPS:+trainer.max_steps="$MAX_STEPS"}

set +x
trap - EXIT

_ray_session_logs="${_ray_tmpdir}/session_latest/logs"
if [[ -d "${_ray_session_logs}" ]]; then
  echo "[gseg] Copying Ray session logs -> ${_ckpt_dir}/ray_logs"
  cp -r "${_ray_session_logs}" "${_ckpt_dir}/ray_logs" 2>/dev/null || true
fi
echo "[gseg] Done. Run log: ${_run_log}"
echo "[gseg] Done. Summary: ${_summary_log}"
