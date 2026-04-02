#!/usr/bin/env bash
# =============================================================
# launch_seg_train.sh — Seg-AOT 消融实验统一训练入口
#
# 调用前必须在各实验脚本里已 source common.sh 并设置:
#   EXP_NAME       实验名称
#   SEG_TASKS      空格分隔的任务列表 (phase_v2t phase_t2v action_v2t action_t2v event_v2t event_t2v)
# =============================================================
set -euo pipefail
set -x

if [[ -z "${EXP_NAME:-}" ]]; then echo "[seg-aot] EXP_NAME not set" >&2; exit 1; fi
if [[ -z "${SEG_TASKS:-}" ]]; then echo "[seg-aot] SEG_TASKS not set" >&2; exit 1; fi

DATA_DIR="${DATA_DIR:-${SEG_AOT_DATA_ROOT}/${EXP_NAME}}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.jsonl}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/val.jsonl}"

# ---- Step 0: 视频切分（clips 不存在时自动触发）----
SKIP_CLIPS="${SKIP_CLIPS:-false}"
if [[ "${SKIP_CLIPS}" != "true" ]]; then
  _need_clips=false
  for _lvl in L1 L2 L3; do
    _dir="${CLIP_ROOT}/${_lvl}"
    if [[ ! -d "${_dir}" ]] || [[ -z "$(ls -A "${_dir}" 2>/dev/null)" ]]; then
      _need_clips=true; break
    fi
  done
  if [[ "${_need_clips}" == "true" ]]; then
    echo "[seg-aot] Clip dirs incomplete, running prepare_all_clips.py ..."
    bash "${SCRIPT_DIR}/prepare_clips.sh"
  fi
fi

# ---- Step A: 数据构建（首次自动触发）----
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[seg-aot] Building data for tasks: ${SEG_TASKS} ..."
  mkdir -p "${DATA_DIR}"
  # shellcheck disable=SC2086
  python3 "${REPO_ROOT}/proxy_data/youcook2_seg/temporal_aot/build_aot_from_seg.py" \
    --annotation-dir "${ANNOTATION_DIR}" \
    --clip-dir-l1 "${CLIP_DIR_L1}" \
    --clip-dir "${CLIP_ROOT}" \
    --output-dir "${DATA_DIR}" \
    --tasks ${SEG_TASKS} \
    --l1-fps "${L1_FPS}" \
    --min-phases "${MIN_PHASES}" \
    --min-events "${MIN_EVENTS}" \
    --min-actions "${MIN_ACTIONS}" \
    --total-val "${TOTAL_VAL}" \
    --train-per-task "${TRAIN_PER_TASK}" \
    --seed 42 \
    --complete-only
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[seg-aot] ERROR: TRAIN_FILE not found after build: ${TRAIN_FILE}" >&2
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

# ---- 自动计算 task weights ----
TASK_WEIGHTS=$(python3 -c "
import json, sys
from collections import Counter
types = Counter()
with open('${TRAIN_FILE}') as f:
    for line in f:
        types[json.loads(line)['problem_type']] += 1
n = len(types)
if n == 0:
    sys.exit(1)
w = {t: round(1.0/n, 4) for t in sorted(types)}
print(json.dumps(w))
")
echo "[seg-aot] Task weights: ${TASK_WEIGHTS}"

# ---- Step B: 训练 ----
echo "[seg-aot] Starting training: ${EXP_NAME}"

TENSORBOARD_DIR="${CHECKPOINT_ROOT}/tensorboard" \
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
  algorithm.online_filtering="${ONLINE_FILTERING}" \
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
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_freq="${VAL_FREQ}" \
  trainer.val_generations_to_log=4 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.logger="[file,tensorboard]" \
  trainer.save_checkpoint_path="${CHECKPOINT_ROOT}/${EXP_NAME}" \
  data.val_batch_size=8

echo "[seg-aot] Done. Log: ${_run_log}"
