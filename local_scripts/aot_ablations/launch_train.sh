#!/usr/bin/env bash
# =============================================================
# launch_train.sh — 完整流程（MCQ 构造 → 混合 → 离线筛选 → 训练）
#
# 调用前必须在各实验脚本里已设置（并 source common.sh）：
#   EXP_NAME         实验名称
#   DATA_DIR         本次实验的数据根目录（所有产出都展开到这里）
#   V2T_OUTPUT       二选一 V2T 输出路径（空 = 不生成）
#   T2V_OUTPUT       二选一 T2V 输出路径（空 = 不生成）
#   THREEWAY_V2T_OUTPUT  3-way V2T 输出路径（空 = 不生成）
#   THREEWAY_T2V_OUTPUT  3-way T2V 输出路径（空 = 不生成）
# =============================================================
set -euo pipefail

# ---- 前置检查 ----
if [[ -z "${EXP_NAME:-}" ]];        then echo "[aot] EXP_NAME not set"        >&2; exit 1; fi
if [[ -z "${DATA_DIR:-}" ]];         then echo "[aot] DATA_DIR not set"         >&2; exit 1; fi
if [[ -z "${CHECKPOINT_ROOT:-}" ]];  then echo "[aot] CHECKPOINT_ROOT not set"  >&2; exit 1; fi
mkdir -p "${DATA_DIR}"

# =========================================================
# 运行日志 & Ray 会话目录：每次执行自动写到实验 ckpt 目录下
# =========================================================
_ckpt_dir="${CHECKPOINT_ROOT}/${EXP_NAME}"
mkdir -p "${_ckpt_dir}"
_ts="$(date +%Y%m%d_%H%M%S)"
_run_log="${_ckpt_dir}/run_${_ts}.log"
_summary_log="${_ckpt_dir}/summary_${_ts}.log"
# 完整日志写入 run_*.log；同时过滤出 [aot]/Error/Traceback 写入 summary_*.log
exec > >(tee >(grep --line-buffered '\[aot\]\|\(Error\|ERROR\|Traceback\|FAILED\)' >> "${_summary_log}") -a "${_run_log}") 2>&1
echo "[aot] ============================================================"
echo "[aot] EXP     : ${EXP_NAME}"
echo "[aot] Date    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "[aot] Log     : ${_run_log}"
echo "[aot] Summary : ${_summary_log}"
echo "[aot] ============================================================"

# Ray 需要本地快速 FS（IPC socket / SHM 不能放 NFS），用实验名命名方便识别
# 训练结束后 Ray session 日志会被复制到 ckpt 目录
# 注意：AF_UNIX socket 路径上限 107 字节，Ray 内部再拼 ~68 字节，
# 所以 _ray_tmpdir 必须 ≤ 39 字节；用 expN 短标识代替完整实验名。
_exp_short="$(echo "${EXP_NAME}" | grep -oP 'exp\d+' | head -1)"
_ray_tmpdir="/tmp/ray_${_exp_short:-${EXP_NAME:0:20}}"
mkdir -p "${_ray_tmpdir}"
export RAY_TMPDIR="${_ray_tmpdir}"
echo "[aot] Ray tmpdir (local): ${_ray_tmpdir}"

MIXED_TRAIN="${DATA_DIR}/mixed_train.jsonl"
MIXED_VAL="${DATA_DIR}/mixed_val.jsonl"
FILTERED_TRAIN="${DATA_DIR}/mixed_train.offline_filtered.jsonl"
FILTER_REPORT="${DATA_DIR}/offline_filter_report.jsonl"

# 若外部未指定 TEST_FILE，使用本实验切分的 val 集
TEST_FILE="${TEST_FILE:-${MIXED_VAL}}"

# FORCE_BUILD 级联到下游：重建数据必然需要重新过滤
if [[ "${FORCE_BUILD:-false}" == "true" ]]; then
  FORCE_FILTER=true
fi

# =========================================================
# Step A: MCQ 构造（build_aot_mcq.py）
# 只当全局 mixed_train.jsonl 不存在时执行
# =========================================================
if [[ ! -f "${MIXED_TRAIN}" || "${FORCE_BUILD:-false}" == "true" ]]; then
  echo "[aot] Building MCQ data for ${EXP_NAME} ..."

  # 收集 build_aot_mcq.py 的 --*-output 参数
  MCQ_ARGS=()
  [[ -n "${V2T_OUTPUT:-}"          ]] && MCQ_ARGS+=(--v2t-output          "${V2T_OUTPUT}")
  [[ -n "${T2V_OUTPUT:-}"          ]] && MCQ_ARGS+=(--t2v-output          "${T2V_OUTPUT}")
  [[ -n "${THREEWAY_V2T_OUTPUT:-}"  ]] && MCQ_ARGS+=(--threeway-output     "${THREEWAY_V2T_OUTPUT}")
  [[ -n "${THREEWAY_T2V_OUTPUT:-}"  ]] && MCQ_ARGS+=(--threeway-t2v-output "${THREEWAY_T2V_OUTPUT}")

  python3 "${REPO_ROOT}/proxy_data/temporal_aot/build_aot_mcq.py" \
    --manifest-jsonl    "${MANIFEST_JSONL}" \
    --caption-pairs     "${CAPTION_PAIRS}" \
    --max-samples       "${MCQ_MAX_SAMPLES}" \
    --min-confidence    "${MCQ_MIN_CONFIDENCE}" \
    "${MCQ_ARGS[@]}"

  # 若 THREEWAY_V2T_FWD_ONLY=true，则只保留 video_direction=="forward" 的 3-way 样本
  # 这样可以单独测试"增加 distractor 难度"而不混入 shuffle-video 信号（exp7 使用）
  if [[ "${THREEWAY_V2T_FWD_ONLY:-false}" == "true" && -n "${THREEWAY_V2T_OUTPUT:-}" && -f "${THREEWAY_V2T_OUTPUT}" ]]; then
    _fwonly="${THREEWAY_V2T_OUTPUT%.jsonl}_fwd_only.jsonl"
    python3 -c "
import json, sys
with open('${THREEWAY_V2T_OUTPUT}') as _f, open('${_fwonly}', 'w') as _o:
    for _line in _f:
        _r = json.loads(_line)
        if _r.get('metadata', {}).get('video_direction') == 'forward':
            _o.write(_line)
"
    _kept=$(wc -l < "${_fwonly}")
    echo "[aot] 3-way V2T fwd-only filter: kept ${_kept} / $(wc -l < "${THREEWAY_V2T_OUTPUT}") samples"
    THREEWAY_V2T_OUTPUT="${_fwonly}"
  fi

  # Step B: 构建训练/验证集
  # 注意: binary 和 3-way 各自独立通道，不再相互合并
  # 各实验脚本通过设置哪些 *_OUTPUT 变量非空来控制使用哪些任务类型
  if [[ -n "${SEG_JSONL:-}" && -f "${SEG_JSONL}" ]]; then
    echo "[aot] Mixing with temporal_seg ..."
    MIX_ARGS=(
      --seg-jsonl        "${SEG_JSONL}"
      --train-output     "${MIXED_TRAIN}"
      --val-output       "${MIXED_VAL}"
      --train-per-source "${MIX_TRAIN_PER_SOURCE:-400}"
      --val-per-source   "${MIX_VAL_PER_SOURCE:-30}"
      --seed             42
    )
    [[ -n "${V2T_OUTPUT:-}" && -f "${V2T_OUTPUT}" ]] && MIX_ARGS+=(--v2t-jsonl "${V2T_OUTPUT}")
    [[ -n "${T2V_OUTPUT:-}" && -f "${T2V_OUTPUT}" ]] && MIX_ARGS+=(--t2v-jsonl "${T2V_OUTPUT}")
    # 3-way 数据通过追加到临时文件后再传入 mix 脚本（或直接 cat 到 train 集）
    if [[ -n "${THREEWAY_V2T_OUTPUT:-}" && -f "${THREEWAY_V2T_OUTPUT}" && -n "${V2T_OUTPUT:-}" ]]; then
      cat "${THREEWAY_V2T_OUTPUT}" >> "${V2T_OUTPUT}"
    fi
    if [[ -n "${THREEWAY_T2V_OUTPUT:-}" && -f "${THREEWAY_T2V_OUTPUT}" && -n "${T2V_OUTPUT:-}" ]]; then
      cat "${THREEWAY_T2V_OUTPUT}" >> "${T2V_OUTPUT}"
    fi

    python3 "${REPO_ROOT}/proxy_data/temporal_aot/mix_aot_with_youcook2.py" "${MIX_ARGS[@]}"
  else
    # 纯 AoT MCQ 模式：不混合 temporal_seg，直接合并 MCQ 输出
    echo "[aot] Pure AoT mode (no temporal_seg) — concatenating MCQ outputs ..."
    : > "${MIXED_TRAIN}.all"
    [[ -n "${V2T_OUTPUT:-}"           && -f "${V2T_OUTPUT}"           ]] && cat "${V2T_OUTPUT}"           >> "${MIXED_TRAIN}.all"
    [[ -n "${T2V_OUTPUT:-}"           && -f "${T2V_OUTPUT}"           ]] && cat "${T2V_OUTPUT}"           >> "${MIXED_TRAIN}.all"
    [[ -n "${THREEWAY_V2T_OUTPUT:-}" && -f "${THREEWAY_V2T_OUTPUT}" ]] && cat "${THREEWAY_V2T_OUTPUT}" >> "${MIXED_TRAIN}.all"
    [[ -n "${THREEWAY_T2V_OUTPUT:-}" && -f "${THREEWAY_T2V_OUTPUT}" ]] && cat "${THREEWAY_T2V_OUTPUT}" >> "${MIXED_TRAIN}.all"
    # 随机打乱后切出 5% 做验证集
    _total=$(wc -l < "${MIXED_TRAIN}.all")
    _val_n=$(python3 -c "print(max(10, int(${_total} * 0.05)))")
    shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:42 -nosalt </dev/zero 2>/dev/null) \
      "${MIXED_TRAIN}.all" -o "${MIXED_TRAIN}.all"
    tail -n "${_val_n}" "${MIXED_TRAIN}.all" > "${MIXED_VAL}"
    head -n $(( _total - _val_n )) "${MIXED_TRAIN}.all" > "${MIXED_TRAIN}"
    rm -f "${MIXED_TRAIN}.all"
  fi

  TRAIN_TOTAL=$(wc -l < "${MIXED_TRAIN}")
  echo "[aot] Mixed train: ${MIXED_TRAIN} (${TRAIN_TOTAL} samples)"
else
  echo "[aot] Reusing existing ${MIXED_TRAIN} (delete to rebuild)"
  # 确保 val 文件也存在（防止只删 val 不删 train 的情况）
  if [[ ! -f "${MIXED_VAL}" ]]; then
    echo "[aot] ERROR: ${MIXED_TRAIN} exists but ${MIXED_VAL} is missing. Delete ${MIXED_TRAIN} and rerun, or set FORCE_BUILD=true." >&2
    exit 1
  fi
fi

# =========================================================
# Step C: 离线 rollout 筛选（多卡数据并行）
# =========================================================
if [[ ! -f "${FILTERED_TRAIN}" || "${FORCE_FILTER:-false}" == "true" ]]; then
  _num_filter_gpus="${FILTER_NUM_GPUS:-${N_GPUS_PER_NODE:-8}}"
  # 每个 shard 占 FILTER_TP_SIZE 张卡；总并行 worker 数 = _num_filter_gpus / FILTER_TP_SIZE
  _tp="${FILTER_TP_SIZE:-1}"
  _num_workers=$(( _num_filter_gpus / _tp ))
  if (( _num_workers < 1 )); then _num_workers=1; fi

  echo "[aot] Running offline rollout filter -> ${FILTERED_TRAIN} (${_num_workers} workers × TP${_tp})"

  _shard_dir="${DATA_DIR}/.filter_shards"
  mkdir -p "${_shard_dir}"
  _pids=()

  for (( _w=0; _w < _num_workers; _w++ )); do
    # 计算该 worker 使用的 GPU ID 列表
    _gpu_start=$(( _w * _tp ))
    _gpu_ids="$(python3 -c "print(','.join(str($_gpu_start + i) for i in range($_tp)))")"

    _shard_output="${_shard_dir}/shard_${_w}.jsonl"
    _shard_report="${_shard_dir}/shard_${_w}_report.jsonl"
    # 给每个 worker 独立的 TMPDIR，防止 vLLM V1 EngineCore 的 ZMQ IPC socket 互相冲突
    # 路径必须短于 ~60 字符，因为 Unix socket 路径上限 107 字符，vLLM 还要拼 UUID
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
    echo "[aot]   worker ${_w}: GPU=${_gpu_ids}, pid=${_pids[-1]}"
    # 错开启动，避免所有 worker 同时加载模型到 CPU 内存导致 OOM
    if (( _w < _num_workers - 1 )); then
      sleep 15
    fi
  done

  # 等待所有 worker 完成，同时轮询 report 文件行数显示汇总进度
  _total_samples=$(wc -l < "${MIXED_TRAIN}")
  _expected_per_shard=$(( (_total_samples + _num_workers - 1) / _num_workers ))
  _expected_total=$(( _expected_per_shard * _num_workers ))
  (( _expected_total > _total_samples )) && _expected_total=${_total_samples}

  # 后台进度监控（每 10 秒刷新一次）
  (
    while true; do
      _done=0
      for (( _i=0; _i < _num_workers; _i++ )); do
        _f="${_shard_dir}/shard_${_i}_report.jsonl"
        [[ -f "${_f}" ]] && _done=$(( _done + $(wc -l < "${_f}") ))
      done
      _pct=$(( _done * 100 / (_expected_total > 0 ? _expected_total : 1) ))
      # 简单 ASCII 进度条（40 格）
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
      echo "[aot] ERROR: worker ${_w} (pid=${_pids[${_w}]}) failed. Log:" >&2
      tail -20 "${_shard_dir}/shard_${_w}.log" >&2
      _any_failed=true
    fi
  done

  # 停止监控，打印最终完成行
  kill "${_monitor_pid}" 2>/dev/null || true
  wait "${_monitor_pid}" 2>/dev/null || true
  _done_final=0
  for (( _i=0; _i < _num_workers; _i++ )); do
    _f="${_shard_dir}/shard_${_i}_report.jsonl"
    [[ -f "${_f}" ]] && _done_final=$(( _done_final + $(wc -l < "${_f}") ))
  done
  printf "\r[offline_filter] [%s] %d/%d (100%%) done\n" "$(printf '%0.s#' $(seq 1 40))" "${_done_final}" "${_expected_total}" >&2
  if [[ "${_any_failed}" == "true" ]]; then
    echo "[aot] Some filter workers failed. Aborting." >&2
    exit 1
  fi

  # 合并所有 shard 输出（仅 shard_N.jsonl，不含 report）
  : > "${FILTERED_TRAIN}"
  : > "${FILTER_REPORT}"
  for (( _w=0; _w < _num_workers; _w++ )); do
    cat "${_shard_dir}/shard_${_w}.jsonl" >> "${FILTERED_TRAIN}"
    cat "${_shard_dir}/shard_${_w}_report.jsonl" >> "${FILTER_REPORT}"
  done
  echo "[aot] All ${_num_workers} shards merged -> ${FILTERED_TRAIN}"
else
  echo "[aot] Reusing filtered file: ${FILTERED_TRAIN} (set FORCE_FILTER=true to redo)"
fi

FILTERED_TOTAL=$(wc -l < "${FILTERED_TRAIN}")
RAW_TOTAL=$(wc -l < "${MIXED_TRAIN}")
echo "[aot] Filtered: ${FILTERED_TOTAL}/${RAW_TOTAL} samples kept -> ${FILTERED_TRAIN}"

# =========================================================
# Step C'/C+: 难度优先采样 + 答案重平衡
# 当 SKIP_CURATE=true 时跳过（在线过滤模式下动态筛选，无需预采样）
# =========================================================
if [[ "${SKIP_CURATE:-false}" == "true" ]]; then
  echo "[aot] SKIP_CURATE=true → using all filtered samples directly (${FILTERED_TOTAL} samples)"
  TRAIN_FILE="${FILTERED_TRAIN}"
else
  # Step C': 难度优先采样，统一到固定样本数（curate）
  # 从 offline_filter_report 中按 mean_reward 分层采样:
  #   medium [0.3, 0.7] 60% / hard (0.0, 0.3) 30% / easy (0.7, 1.0) 10%
  #   排除 mean_reward==0（base 完全不会）和 ==1（已被 offline filter 删）
  CURATED_TRAIN="${DATA_DIR}/mixed_train.curated_${CURATE_TARGET_COUNT}.jsonl"
  if [[ ! -f "${CURATED_TRAIN}" || "${FORCE_CURATE:-false}" == "true" || "${FORCE_FILTER:-false}" == "true" ]]; then
    echo "[aot] Curating ${CURATE_TARGET_COUNT} samples -> ${CURATED_TRAIN}"

    CURATE_ARGS=(
      --report-jsonl   "${CURATE_REPORT_JSONLS:-${FILTER_REPORT}}"
      --train-jsonl    "${CURATE_TRAIN_JSONLS:-${MIXED_TRAIN}}"
      --output-jsonl   "${CURATED_TRAIN}"
      --target-count   "${CURATE_TARGET_COUNT}"
      --mid-ratio      "${CURATE_MID_RATIO}"
      --hard-ratio     "${CURATE_HARD_RATIO}"
      --easy-ratio     "${CURATE_EASY_RATIO}"
      --mid-lo         "${CURATE_MID_LO}"
      --mid-hi         "${CURATE_MID_HI}"
      --seed           42
    )
    if [[ -n "${CURATE_PER_TYPE_QUOTA:-}" ]]; then
      CURATE_ARGS+=(--per-type-quota "${CURATE_PER_TYPE_QUOTA}")
    fi

    python3 "${SCRIPT_DIR}/curate_1k_samples.py" "${CURATE_ARGS[@]}"
  else
    echo "[aot] Reusing curated file: ${CURATED_TRAIN} (set FORCE_CURATE=true to redo)"
  fi

  CURATED_TOTAL=$(wc -l < "${CURATED_TRAIN}")
  echo "[aot] Curated: ${CURATED_TOTAL}/${FILTERED_TOTAL} samples selected -> ${CURATED_TRAIN}"

  # Step C+: 答案选项重平衡（binary A/B + 3-way A/B/C）
  # 消除离线过滤后因位置偏差导致的答案分布不均
  BALANCED_TRAIN="${DATA_DIR}/mixed_train.curated_${CURATE_TARGET_COUNT}.balanced.jsonl"
  if [[ ! -f "${BALANCED_TRAIN}" || "${FORCE_CURATE:-false}" == "true" || "${FORCE_FILTER:-false}" == "true" ]]; then
    echo "[aot] Rebalancing answer distribution -> ${BALANCED_TRAIN}"
    python3 "${REPO_ROOT}/proxy_data/temporal_aot/rebalance_aot_answers.py" \
      --input-jsonl  "${CURATED_TRAIN}" \
      --output-jsonl "${BALANCED_TRAIN}" \
      --problem-types "aot_v2t,aot_t2v,aot_3way_v2t,aot_3way_t2v" \
      --balance-scope problem_type \
      --seed 42
  else
    echo "[aot] Reusing balanced file: ${BALANCED_TRAIN} (set FORCE_CURATE=true to redo)"
  fi

  TRAIN_FILE="${BALANCED_TRAIN}"
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
  echo "[aot] Copying Ray session logs -> ${_ckpt_dir}/ray_logs"
  cp -r "${_ray_session_logs}" "${_ckpt_dir}/ray_logs" 2>/dev/null || \
    echo "[aot] Warning: Ray log copy failed (non-fatal)" >&2
else
  echo "[aot] Ray session logs not found at ${_ray_session_logs} (skipping)"
fi
echo "[aot] Done. Run log: ${_run_log}  |  Summary: ${_summary_log}"
