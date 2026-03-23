#!/usr/bin/env bash
# =============================================================
# launch_train_cross_exp.sh — 跨实验混合流程
#
# 用于 exp7/exp8 等混合实验：从多个已完成实验的 report 中
# 按 per-type-quota 采样 → 重平衡 → 训练
#
# 跳过 Step A/B/C（MCQ 构造、混合、离线筛选），直接从 Step C' 开始。
# 前提：所引用的实验必须已完成离线筛选（report 文件存在）。
#
# 调用前必须设置：
#   EXP_NAME              实验名称
#   DATA_DIR              本次实验的数据目录
#   CURATE_REPORT_JSONLS  逗号分隔的 report 文件路径
#   CURATE_TRAIN_JSONLS   逗号分隔的 train 文件路径（与 report 对齐）
#   CURATE_PER_TYPE_QUOTA JSON dict，各 problem_type 的目标数量
# =============================================================
set -euo pipefail

# ---- 前置检查 ----
if [[ -z "${EXP_NAME:-}" ]];              then echo "[aot] EXP_NAME not set"              >&2; exit 1; fi
if [[ -z "${DATA_DIR:-}" ]];               then echo "[aot] DATA_DIR not set"               >&2; exit 1; fi
if [[ -z "${CHECKPOINT_ROOT:-}" ]];        then echo "[aot] CHECKPOINT_ROOT not set"        >&2; exit 1; fi
if [[ -z "${CURATE_REPORT_JSONLS:-}" ]];   then echo "[aot] CURATE_REPORT_JSONLS not set"   >&2; exit 1; fi
if [[ -z "${CURATE_TRAIN_JSONLS:-}" ]];    then echo "[aot] CURATE_TRAIN_JSONLS not set"    >&2; exit 1; fi
if [[ -z "${CURATE_PER_TYPE_QUOTA:-}" ]];  then echo "[aot] CURATE_PER_TYPE_QUOTA not set"  >&2; exit 1; fi
mkdir -p "${DATA_DIR}"

# ---- 检查依赖文件是否存在 ----
IFS=',' read -ra _report_files <<< "${CURATE_REPORT_JSONLS}"
IFS=',' read -ra _train_files  <<< "${CURATE_TRAIN_JSONLS}"
for _f in "${_report_files[@]}"; do
  _f="${_f## }"; _f="${_f%% }"  # trim spaces
  if [[ ! -f "${_f}" ]]; then
    echo "[aot] ERROR: prerequisite report not found: ${_f}" >&2
    echo "[aot] Please run the source experiments first." >&2
    exit 1
  fi
done
for _f in "${_train_files[@]}"; do
  _f="${_f## }"; _f="${_f%% }"
  if [[ ! -f "${_f}" ]]; then
    echo "[aot] ERROR: prerequisite train file not found: ${_f}" >&2
    echo "[aot] Please run the source experiments first." >&2
    exit 1
  fi
done

# =========================================================
# 运行日志 & Ray 会话目录
# =========================================================
_ckpt_dir="${CHECKPOINT_ROOT}/${EXP_NAME}"
mkdir -p "${_ckpt_dir}"
_run_log="${_ckpt_dir}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${_run_log}") 2>&1
echo "[aot] ============================================================"
echo "[aot] EXP : ${EXP_NAME} (cross-experiment mixed)"
echo "[aot] Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "[aot] Log : ${_run_log}"
echo "[aot] Reports: ${CURATE_REPORT_JSONLS}"
echo "[aot] Trains : ${CURATE_TRAIN_JSONLS}"
echo "[aot] Quota  : ${CURATE_PER_TYPE_QUOTA}"
echo "[aot] ============================================================"

_ray_tmpdir="/tmp/ray_${EXP_NAME}"
mkdir -p "${_ray_tmpdir}"
export RAY_TMPDIR="${_ray_tmpdir}"

# 若外部未指定 TEST_FILE，使用第一个源实验的 val
MIXED_VAL="${DATA_DIR}/mixed_val.jsonl"
TEST_FILE="${TEST_FILE:-${MIXED_VAL}}"
# 若 val 不存在，从第一个 train 的同目录下找
if [[ ! -f "${TEST_FILE}" ]]; then
  _first_train="${_train_files[0]}"
  _first_train="${_first_train## }"; _first_train="${_first_train%% }"
  _first_dir="$(dirname "${_first_train}")"
  if [[ -f "${_first_dir}/mixed_val.jsonl" ]]; then
    TEST_FILE="${_first_dir}/mixed_val.jsonl"
    echo "[aot] Using val from source exp: ${TEST_FILE}"
  fi
fi

# =========================================================
# Step C': 难度优先采样（从跨实验 report 中采样）
# =========================================================
CURATED_TRAIN="${DATA_DIR}/mixed_train.curated_${CURATE_TARGET_COUNT}.jsonl"
if [[ ! -f "${CURATED_TRAIN}" || "${FORCE_CURATE:-false}" == "true" ]]; then
  echo "[aot] Curating ${CURATE_TARGET_COUNT} samples -> ${CURATED_TRAIN}"

  python3 "${SCRIPT_DIR}/curate_1k_samples.py" \
    --report-jsonl   "${CURATE_REPORT_JSONLS}" \
    --train-jsonl    "${CURATE_TRAIN_JSONLS}" \
    --output-jsonl   "${CURATED_TRAIN}" \
    --target-count   "${CURATE_TARGET_COUNT}" \
    --mid-ratio      "${CURATE_MID_RATIO}" \
    --hard-ratio     "${CURATE_HARD_RATIO}" \
    --easy-ratio     "${CURATE_EASY_RATIO}" \
    --mid-lo         "${CURATE_MID_LO}" \
    --mid-hi         "${CURATE_MID_HI}" \
    --per-type-quota "${CURATE_PER_TYPE_QUOTA}" \
    --seed           42
else
  echo "[aot] Reusing curated file: ${CURATED_TRAIN} (set FORCE_CURATE=true to redo)"
fi

CURATED_TOTAL=$(wc -l < "${CURATED_TRAIN}")
echo "[aot] Curated: ${CURATED_TOTAL} samples -> ${CURATED_TRAIN}"

# =========================================================
# Step C+: 答案选项重平衡
# =========================================================
BALANCED_TRAIN="${DATA_DIR}/mixed_train.curated_${CURATE_TARGET_COUNT}.balanced.jsonl"
if [[ ! -f "${BALANCED_TRAIN}" || "${FORCE_CURATE:-false}" == "true" ]]; then
  echo "[aot] Rebalancing answer distribution -> ${BALANCED_TRAIN}"
  python3 "${REPO_ROOT}/proxy_data/temporal_aot/rebalance_aot_answers.py" \
    --input-jsonl  "${CURATED_TRAIN}" \
    --output-jsonl "${BALANCED_TRAIN}" \
    --problem-types "aot_v2t,aot_t2v,aot_3way_v2t,aot_3way_t2v" \
    --balance-scope problem_type \
    --seed 42
else
  echo "[aot] Reusing balanced file: ${BALANCED_TRAIN}"
fi

TRAIN_FILE="${BALANCED_TRAIN}"

# =========================================================
# Step D: 自动计算任务权重
# =========================================================
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
        if not line: continue
        obj = json.loads(line)
        t = str(obj.get("problem_type") or "").strip()
        if t: counter[t] += 1
if not counter: raise SystemExit(f"No problem_type in {path}")
task_names = sorted(counter)
if mode in {"equal","uniform"}:
    w = {t: 1.0/len(task_names) for t in task_names}
else:
    total = sum(counter.values())
    w = {t: counter[t]/total for t in task_names}
summary = ", ".join(f"{t}={counter[t]}" for t in task_names)
sys.stderr.write(f"Task counts: {summary}\n")
print(json.dumps(w, ensure_ascii=False, separators=(",",":")))
PY
)"
fi

# =========================================================
# Step E: 启动训练
# =========================================================
echo "[aot] Starting training: ${EXP_NAME}"
echo "[aot] TRAIN_FILE : ${TRAIN_FILE}"
echo "[aot] LR=${LR}  warmup_style=${WARMUP_STYLE}  warmup_ratio=${LR_WARMUP_RATIO}  min_lr_ratio=${LR_MIN_RATIO}"

set -x
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
  data.task_homogeneous_batching="${TASK_HOMOGENEOUS}" \
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
  trainer.val_freq="${VAL_FREQ}" \
  trainer.val_generations_to_log=4 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.logger="[file,tensorboard]" \
  trainer.save_checkpoint_path="${CHECKPOINT_ROOT}/${EXP_NAME}" \
  data.val_batch_size=8 ${MAX_STEPS:+trainer.max_steps="$MAX_STEPS"}

# =========================================================
# 训练结束：将 Ray session 日志复制到 ckpt 目录
# =========================================================
set +x
_ray_session_logs="${_ray_tmpdir}/session_latest/logs"
if [[ -d "${_ray_session_logs}" ]]; then
  echo "[aot] Copying Ray session logs -> ${_ckpt_dir}/ray_logs"
  cp -r "${_ray_session_logs}" "${_ckpt_dir}/ray_logs" 2>/dev/null || \
    echo "[aot] Warning: Ray log copy failed (non-fatal)" >&2
else
  echo "[aot] Ray session logs not found at ${_ray_session_logs} (skipping)"
fi
echo "[aot] Done. Run log: ${_run_log}"
