#!/usr/bin/env bash
# =============================================================
# launch_sort_train.sh — Event Shuffle (Sort) 实验训练入口
#
# 流程: build_event_shuffle.py 构建数据 → 训练
#
# 调用前必须 source common.sh 并设置:
#   EXP_NAME       实验名称
#   DATA_DIR        数据输出目录
# =============================================================
set -euo pipefail

if [[ -z "${EXP_NAME:-}" ]];   then echo "[el] EXP_NAME not set"   >&2; exit 1; fi
if [[ -z "${DATA_DIR:-}" ]];   then echo "[el] DATA_DIR not set"   >&2; exit 1; fi
mkdir -p "${DATA_DIR}"

# =========================================================
# 运行日志 & Ray 会话目录
# =========================================================
_ckpt_dir="${CHECKPOINT_ROOT}/${EXP_NAME}"
mkdir -p "${_ckpt_dir}"
_ts="$(date +%Y%m%d_%H%M%S)"
_run_log="${_ckpt_dir}/run_${_ts}.log"
_summary_log="${_ckpt_dir}/summary_${_ts}.log"
exec > >(tee >(grep --line-buffered -E '^\[el\]|ERROR|Error|Exception|Traceback|SIGTERM|OOM|step [0-9]|reward|val.*metric' >> "${_summary_log}") -a "${_run_log}") 2>&1
echo "[el] ============================================================"
echo "[el] EXP    : ${EXP_NAME}"
echo "[el] Date   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "[el] Log    : ${_run_log}"
echo "[el] Summary: ${_summary_log}"
echo "[el] ============================================================"

# Ray tmpdir
_exp_short="$(echo "${EXP_NAME}" | grep -oE 'exp[0-9]+' || echo "${EXP_NAME:0:12}")"
_ray_tmpdir="/tmp/ray_${_exp_short}"
mkdir -p "${_ray_tmpdir}"
export RAY_TMPDIR="${_ray_tmpdir}"

TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

# =========================================================
# Step A: 数据构造（build_event_shuffle.py）
# =========================================================
if [[ ! -f "${TRAIN_FILE}" || "${FORCE_BUILD:-false}" == "true" ]]; then
  echo "[el] Building event shuffle data ..."
  python3 "${REPO_ROOT}/proxy_data/youcook2_seg/event_logic/build_event_shuffle.py" \
    --annotation-dir "${ANNOTATION_DIR}" \
    --clip-dir       "${CLIP_DIR}" \
    --output-dir     "${DATA_DIR}" \
    --level          "${SORT_LEVEL:-l2}" \
    --min-events     "${MIN_EVENTS:-3}" \
    --max-events     "${MAX_EVENTS:-8}" \
    --seq-len        "${SORT_SEQ_LEN:-5}" \
    --samples-per-group "${SAMPLES_PER_GROUP:-1}" \
    --complete-only \
    ${FILTER_ORDER:+--filter-order} \
    --train-budget   "${TRAIN_BUDGET:--1}" \
    --val-count      "${VAL_COUNT:-100}" \
    --seed           "${BUILD_SEED:-42}"
else
  echo "[el] Reusing existing data at ${DATA_DIR}"
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[el] ERROR: TRAIN_FILE not found: ${TRAIN_FILE}" >&2
  exit 1
fi

echo "[el] Train: ${TRAIN_FILE} ($(wc -l < "${TRAIN_FILE}") samples)"
echo "[el] Val:   ${TEST_FILE} ($(wc -l < "${TEST_FILE}") samples)"

# =========================================================
# GPU filler: 保持利用率 ≥ 80%，训练阶段自动暂停
# =========================================================
_filler_script="${REPO_ROOT}/local_scripts/gpu_filler.py"
if [[ "${ENABLE_GPU_FILLER:-true}" == "true" ]] && [[ -f "${_filler_script}" ]]; then
  if pgrep -f "gpu_filler.py" > /dev/null 2>&1; then
    echo "[el] Killing old filler instances..."
    pkill -f "gpu_filler.py" 2>/dev/null || true
    sleep 2
  fi
  echo "[el] Starting GPU filler (target=${FILLER_TARGET_UTIL:-85}%)"
  nohup python3 "${_filler_script}" \
    --target-util "${FILLER_TARGET_UTIL:-85}" \
    --batch "${FILLER_BATCH:-50}" \
    --matrix-size "${FILLER_MATRIX:-8192}" \
    > /tmp/filler.log 2>&1 &
  echo "[el] GPU filler started (PID $!), log: /tmp/filler.log"
fi

# 训练结束只清理信号文件，不杀 filler
trap 'rm -f /tmp/verl_gpu_phase' EXIT

# =========================================================
# Step B: 训练
# =========================================================
echo "[el] Starting training: ${EXP_NAME}"
echo "[el] LR=${LR}  warmup=${WARMUP_STYLE}  warmup_ratio=${LR_WARMUP_RATIO}"

set -x
TENSORBOARD_DIR="${CHECKPOINT_ROOT}/${EXP_NAME}/tensorboard" \
TENSORBOARD_FLAT=1 \
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

# =========================================================
# 训练结束
# =========================================================
set +x
_ray_session_logs="${_ray_tmpdir}/session_latest/logs"
if [[ -d "${_ray_session_logs}" ]]; then
  echo "[el] Copying Ray session logs -> ${_ckpt_dir}/ray_logs"
  cp -r "${_ray_session_logs}" "${_ckpt_dir}/ray_logs" 2>/dev/null || \
    echo "[el] Warning: Ray log copy failed (non-fatal)" >&2
fi
echo "[el] Done. Run log: ${_run_log}"
echo "[el] Done. Summary: ${_summary_log}"
