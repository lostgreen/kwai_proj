#!/usr/bin/env bash
# =============================================================
# launch_train.sh — Event Logic 完整流程
#   数据构造 → 切分 train/val → 离线筛选 → 训练
#
# 调用前必须在各实验脚本里已设置（并 source common.sh）：
#   EXP_NAME          实验名称
#   DATA_DIR          本次实验的数据根目录
#   ADD_PER_VIDEO     每视频 add 样本数（0 = 不生成）
#   REPLACE_PER_VIDEO 每视频 replace 样本数（0 = 不生成）
#   SORT_PER_VIDEO    每视频 sort 样本数（0 = 不生成）
# =============================================================
set -euo pipefail

# ---- 前置检查 ----
if [[ -z "${EXP_NAME:-}" ]];        then echo "[el] EXP_NAME not set"        >&2; exit 1; fi
if [[ -z "${DATA_DIR:-}" ]];         then echo "[el] DATA_DIR not set"         >&2; exit 1; fi
if [[ -z "${CHECKPOINT_ROOT:-}" ]];  then echo "[el] CHECKPOINT_ROOT not set"  >&2; exit 1; fi
mkdir -p "${DATA_DIR}"

# =========================================================
# 运行日志 & Ray 会话目录
# =========================================================
_ckpt_dir="${CHECKPOINT_ROOT}/${EXP_NAME}"
mkdir -p "${_ckpt_dir}"
_ts="$(date +%Y%m%d_%H%M%S)"
_run_log="${_ckpt_dir}/run_${_ts}.log"
_summary_log="${_ckpt_dir}/summary_${_ts}.log"
exec > >(tee >(grep --line-buffered '\[el\]\|\(Error\|ERROR\|Traceback\|FAILED\)' >> "${_summary_log}") -a "${_run_log}") 2>&1
echo "[el] ============================================================"
echo "[el] EXP     : ${EXP_NAME}"
echo "[el] Date    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "[el] Log     : ${_run_log}"
echo "[el] Summary : ${_summary_log}"
echo "[el] ============================================================"

# Ray tmpdir: AF_UNIX socket 路径上限 107 字节
_exp_short="$(echo "${EXP_NAME}" | grep -oP 'exp\d+' | head -1)"
_ray_tmpdir="/tmp/ray_${_exp_short:-${EXP_NAME:0:20}}"
mkdir -p "${_ray_tmpdir}"
export RAY_TMPDIR="${_ray_tmpdir}"
echo "[el] Ray tmpdir (local): ${_ray_tmpdir}"

RAW_OUTPUT="${DATA_DIR}/l2_event_logic_raw.jsonl"
MIXED_TRAIN="${DATA_DIR}/mixed_train.jsonl"
MIXED_VAL="${DATA_DIR}/mixed_val.jsonl"
FILTERED_TRAIN="${DATA_DIR}/mixed_train.offline_filtered.jsonl"
FILTER_REPORT="${DATA_DIR}/offline_filter_report.jsonl"

# 若外部未指定 TEST_FILE，使用本实验切分的 val 集
TEST_FILE="${TEST_FILE:-${MIXED_VAL}}"

# FORCE_BUILD 级联到下游
if [[ "${FORCE_BUILD:-false}" == "true" ]]; then
  FORCE_FILTER=true
fi

# =========================================================
# Step A: 数据构造（build_l2_event_logic.py）
# =========================================================
if [[ ! -f "${MIXED_TRAIN}" || "${FORCE_BUILD:-false}" == "true" ]]; then
  echo "[el] Building Event Logic data for ${EXP_NAME} ..."
  echo "[el]   ADD_PER_VIDEO=${ADD_PER_VIDEO:-0}  REPLACE_PER_VIDEO=${REPLACE_PER_VIDEO:-0}  SORT_PER_VIDEO=${SORT_PER_VIDEO:-0}"

  BUILD_ARGS=(
    --annotation-dir  "${L2_ANNOTATION_DIR}"
    --clips-dir       "${L2_CLIPS_DIR}"
    --frames-dir      "${L2_FRAMES_DIR}"
    --output          "${RAW_OUTPUT}"
    --add-per-video   "${ADD_PER_VIDEO:-0}"
    --replace-per-video "${REPLACE_PER_VIDEO:-0}"
    --sort-per-video  "${SORT_PER_VIDEO:-0}"
    --min-events      "${MIN_EVENTS}"
    --min-context     "${MIN_CONTEXT}"
    --max-context     "${MAX_CONTEXT}"
    --replace-seq-len "${REPLACE_SEQ_LEN}"
    --sort-seq-len    "${SORT_SEQ_LEN}"
    --seed            "${BUILD_SEED}"
    --shuffle
  )

  # AI 因果过滤（可选）
  if [[ "${FILTER_AI:-false}" == "true" ]]; then
    BUILD_ARGS+=(
      --filter
      --api-base "${AI_API_BASE}"
      --model    "${AI_MODEL}"
      --confidence-threshold "${AI_CONFIDENCE}"
      --filter-workers "${AI_FILTER_WORKERS}"
    )
    if [[ -n "${AI_API_KEY:-}" ]]; then
      BUILD_ARGS+=(--api-key "${AI_API_KEY}")
    fi
    echo "[el]   AI Filter: ON (model=${AI_MODEL}, confidence>=${AI_CONFIDENCE})"
  else
    echo "[el]   AI Filter: OFF"
  fi

  python3 "${REPO_ROOT}/proxy_data/event_logic/build_l2_event_logic.py" "${BUILD_ARGS[@]}"

  # Step B: 切分 train/val（5% 验证集）
  echo "[el] Splitting train/val ..."
  _total=$(wc -l < "${RAW_OUTPUT}")
  _val_n=$(python3 -c "print(max(10, int(${_total} * 0.05)))")
  shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:42 -nosalt </dev/zero 2>/dev/null) \
    "${RAW_OUTPUT}" -o "${RAW_OUTPUT}"
  tail -n "${_val_n}" "${RAW_OUTPUT}" > "${MIXED_VAL}"
  head -n $(( _total - _val_n )) "${RAW_OUTPUT}" > "${MIXED_TRAIN}"

  TRAIN_TOTAL=$(wc -l < "${MIXED_TRAIN}")
  VAL_TOTAL=$(wc -l < "${MIXED_VAL}")
  echo "[el] Mixed train: ${MIXED_TRAIN} (${TRAIN_TOTAL} samples)"
  echo "[el] Mixed val  : ${MIXED_VAL} (${VAL_TOTAL} samples)"
else
  echo "[el] Reusing existing ${MIXED_TRAIN} (delete to rebuild)"
  if [[ ! -f "${MIXED_VAL}" ]]; then
    echo "[el] ERROR: ${MIXED_TRAIN} exists but ${MIXED_VAL} is missing. Delete ${MIXED_TRAIN} and rerun, or set FORCE_BUILD=true." >&2
    exit 1
  fi
fi

# =========================================================
# Step C: 离线 rollout 筛选（多卡数据并行）
# =========================================================
if [[ ! -f "${FILTERED_TRAIN}" || "${FORCE_FILTER:-false}" == "true" ]]; then
  _num_filter_gpus="${FILTER_NUM_GPUS:-${N_GPUS_PER_NODE:-8}}"
  _tp="${FILTER_TP_SIZE:-1}"
  _num_workers=$(( _num_filter_gpus / _tp ))
  if (( _num_workers < 1 )); then _num_workers=1; fi

  echo "[el] Running offline rollout filter -> ${FILTERED_TRAIN} (${_num_workers} workers x TP${_tp})"

  _shard_dir="${DATA_DIR}/.filter_shards"
  mkdir -p "${_shard_dir}"
  _pids=()

  for (( _w=0; _w < _num_workers; _w++ )); do
    _gpu_start=$(( _w * _tp ))
    _gpu_ids="$(python3 -c "print(','.join(str($_gpu_start + i) for i in range($_tp)))")"

    _shard_output="${_shard_dir}/shard_${_w}.jsonl"
    _shard_report="${_shard_dir}/shard_${_w}_report.jsonl"
    _worker_tmpdir="/tmp/vllm_filter_${EXP_NAME}_${_w}"
    mkdir -p "${_worker_tmpdir}"

    CUDA_VISIBLE_DEVICES="${_gpu_ids}" \
    TMPDIR="${_worker_tmpdir}" \
    NCCL_SHM_DISABLE=1 \
    python3 "${REPO_ROOT}/local_scripts/offline_rollout_filter.py" \
      --input_jsonl  "${MIXED_TRAIN}" \
      --output_jsonl "${_shard_output}" \
      --report_jsonl "${_shard_report}" \
      --model_path   "${MODEL_PATH}" \
      --reward_function "${REWARD_FUNCTION}" \
      --backend vllm \
      --num_rollouts  "${FILTER_ROLLOUT_N:-${ROLLOUT_N}}" \
      --temperature   0.7 \
      --top_p         0.9 \
      --max_new_tokens "${MAX_RESPONSE_LEN}" \
      --video_fps     "${VIDEO_FPS}" \
      --max_frames    "${MAX_FRAMES}" \
      --min_frames    "${MIN_FRAMES:-0}" \
      --max_pixels    "${MAX_PIXELS}" \
      --min_pixels    "${MIN_PIXELS}" \
      --min_mean_reward "${MIN_MEAN_REWARD:-0.0}" \
      --max_mean_reward "${MAX_MEAN_REWARD:-1.0}" \
      --tensor_parallel_size "${_tp}" \
      --gpu_memory_utilization "${FILTER_GPU_MEM_UTIL:-0.7}" \
      --max_model_len "${FILTER_MAX_MODEL_LEN:-16384}" \
      --shard_id "${_w}" \
      --num_shards "${_num_workers}" \
      > "${_shard_dir}/shard_${_w}.log" 2>&1 &
    _pids+=($!)
    echo "[el]   worker ${_w}: GPU=${_gpu_ids}, pid=${_pids[-1]}"
    if (( _w < _num_workers - 1 )); then
      sleep 15
    fi
  done

  # 等待所有 worker + 进度监控
  _total_samples=$(wc -l < "${MIXED_TRAIN}")
  _expected_per_shard=$(( (_total_samples + _num_workers - 1) / _num_workers ))
  _expected_total=$(( _expected_per_shard * _num_workers ))
  (( _expected_total > _total_samples )) && _expected_total=${_total_samples}

  (
    while true; do
      _done=0
      for (( _i=0; _i < _num_workers; _i++ )); do
        _f="${_shard_dir}/shard_${_i}_report.jsonl"
        [[ -f "${_f}" ]] && _done=$(( _done + $(wc -l < "${_f}") ))
      done
      _pct=$(( _done * 100 / (_expected_total > 0 ? _expected_total : 1) ))
      _filled=$(( _pct * 40 / 100 ))
      _bar="$(printf '%0.s#' $(seq 1 $_filled 2>/dev/null))$(printf '%0.s-' $(seq 1 $(( 40 - _filled )) 2>/dev/null))"
      printf "\r[offline_filter] [%s] %d/%d (%d%%)  " "${_bar}" "${_done}" "${_expected_total}" "${_pct}" >&2
      sleep 10
    done
  ) &
  _monitor_pid=$!

  _any_failed=false
  for (( _w=0; _w < _num_workers; _w++ )); do
    if ! wait "${_pids[${_w}]}"; then
      echo "" >&2
      echo "[el] ERROR: worker ${_w} (pid=${_pids[${_w}]}) failed. Log:" >&2
      tail -20 "${_shard_dir}/shard_${_w}.log" >&2
      _any_failed=true
    fi
  done

  kill "${_monitor_pid}" 2>/dev/null || true
  wait "${_monitor_pid}" 2>/dev/null || true
  _done_final=0
  for (( _i=0; _i < _num_workers; _i++ )); do
    _f="${_shard_dir}/shard_${_i}_report.jsonl"
    [[ -f "${_f}" ]] && _done_final=$(( _done_final + $(wc -l < "${_f}") ))
  done
  printf "\r[offline_filter] [%s] %d/%d (100%%) done\n" "$(printf '%0.s#' $(seq 1 40))" "${_done_final}" "${_expected_total}" >&2
  if [[ "${_any_failed}" == "true" ]]; then
    echo "[el] Some filter workers failed. Aborting." >&2
    exit 1
  fi

  # 合并所有 shard 输出
  : > "${FILTERED_TRAIN}"
  : > "${FILTER_REPORT}"
  for (( _w=0; _w < _num_workers; _w++ )); do
    cat "${_shard_dir}/shard_${_w}.jsonl" >> "${FILTERED_TRAIN}"
    cat "${_shard_dir}/shard_${_w}_report.jsonl" >> "${FILTER_REPORT}"
  done
  echo "[el] All ${_num_workers} shards merged -> ${FILTERED_TRAIN}"
else
  echo "[el] Reusing filtered file: ${FILTERED_TRAIN} (set FORCE_FILTER=true to redo)"
fi

FILTERED_TOTAL=$(wc -l < "${FILTERED_TRAIN}")
RAW_TOTAL=$(wc -l < "${MIXED_TRAIN}")
echo "[el] Filtered: ${FILTERED_TOTAL}/${RAW_TOTAL} samples kept -> ${FILTERED_TRAIN}"

# =========================================================
# Step C': 难度优先采样（可选，默认跳过）
# =========================================================
if [[ "${SKIP_CURATE:-false}" == "true" ]]; then
  echo "[el] SKIP_CURATE=true -> using all filtered samples directly (${FILTERED_TOTAL} samples)"
  TRAIN_FILE="${FILTERED_TRAIN}"
else
  CURATED_TRAIN="${DATA_DIR}/mixed_train.curated_${CURATE_TARGET_COUNT}.jsonl"
  if [[ ! -f "${CURATED_TRAIN}" || "${FORCE_CURATE:-false}" == "true" || "${FORCE_FILTER:-false}" == "true" ]]; then
    echo "[el] Curating ${CURATE_TARGET_COUNT} samples -> ${CURATED_TRAIN}"

    CURATE_ARGS=(
      --report-jsonl   "${CURATE_REPORT_JSONLS:-${FILTER_REPORT}}"
      --train-jsonl    "${CURATE_TRAIN_JSONLS:-${MIXED_TRAIN}}"
      --output-jsonl   "${CURATED_TRAIN}"
      --target-count   "${CURATE_TARGET_COUNT}"
      --mid-ratio      "${CURATE_MID_RATIO:-0.6}"
      --hard-ratio     "${CURATE_HARD_RATIO:-0.3}"
      --easy-ratio     "${CURATE_EASY_RATIO:-0.1}"
      --mid-lo         "${CURATE_MID_LO:-0.3}"
      --mid-hi         "${CURATE_MID_HI:-0.7}"
      --seed           42
    )
    if [[ -n "${CURATE_PER_TYPE_QUOTA:-}" ]]; then
      CURATE_ARGS+=(--per-type-quota "${CURATE_PER_TYPE_QUOTA}")
    fi

    python3 "${SCRIPT_DIR}/curate_1k_samples.py" "${CURATE_ARGS[@]}"
  else
    echo "[el] Reusing curated file: ${CURATED_TRAIN} (set FORCE_CURATE=true to redo)"
  fi

  CURATED_TOTAL=$(wc -l < "${CURATED_TRAIN}")
  echo "[el] Curated: ${CURATED_TOTAL}/${FILTERED_TOTAL} samples selected -> ${CURATED_TRAIN}"
  TRAIN_FILE="${CURATED_TRAIN}"
fi

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
echo "[el] Starting training: ${EXP_NAME}"
echo "[el] TRAIN_FILE : ${TRAIN_FILE}"
echo "[el] LR=${LR}  warmup_style=${WARMUP_STYLE}  warmup_ratio=${LR_WARMUP_RATIO}  min_lr_ratio=${LR_MIN_RATIO}"

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
  data.min_frames="${MIN_FRAMES:-0}" \
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
  echo "[el] Copying Ray session logs -> ${_ckpt_dir}/ray_logs"
  cp -r "${_ray_session_logs}" "${_ckpt_dir}/ray_logs" 2>/dev/null || \
    echo "[el] Warning: Ray log copy failed (non-fatal)" >&2
else
  echo "[el] Ray session logs not found at ${_ray_session_logs} (skipping)"
fi
echo "[el] Done. Run log: ${_run_log}  |  Summary: ${_summary_log}"
