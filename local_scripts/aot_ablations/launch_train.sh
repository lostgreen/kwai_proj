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
if [[ -z "${EXP_NAME:-}" ]]; then echo "[aot] EXP_NAME not set" >&2; exit 1; fi
if [[ -z "${DATA_DIR:-}" ]];  then echo "[aot] DATA_DIR not set"  >&2; exit 1; fi
mkdir -p "${DATA_DIR}"

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
# Step C: 离线 rollout 筛选
# =========================================================
if [[ ! -f "${FILTERED_TRAIN}" || "${FORCE_FILTER:-false}" == "true" ]]; then
  echo "[aot] Running offline rollout filter -> ${FILTERED_TRAIN}"
  python3 "${REPO_ROOT}/local_scripts/offline_rollout_filter.py" \
    --input_jsonl  "${MIXED_TRAIN}" \
    --output_jsonl "${FILTERED_TRAIN}" \
    --report_jsonl "${FILTER_REPORT}" \
    --model_path   "${MODEL_PATH}" \
    --reward_function "${REWARD_FUNCTION}" \
    --backend vllm \
    --num_rollouts  "${ROLLOUT_N}" \
    --temperature   0.7 \
    --top_p         0.9 \
    --max_new_tokens "${MAX_RESPONSE_LEN}" \
    --video_fps     "${VIDEO_FPS}" \
    --max_frames    "${MAX_FRAMES}" \
    --max_pixels    "${MAX_PIXELS}" \
    --min_pixels    "${MIN_PIXELS}" \
    --tensor_parallel_size "${FILTER_TP_SIZE:-1}" \
    --gpu_memory_utilization "${FILTER_GPU_MEM_UTIL:-0.7}" \
    --max_model_len "${FILTER_MAX_MODEL_LEN:-16384}"
else
  echo "[aot] Reusing filtered file: ${FILTERED_TRAIN} (set FORCE_FILTER=true to redo)"
fi

FILTERED_TOTAL=$(wc -l < "${FILTERED_TRAIN}")
RAW_TOTAL=$(wc -l < "${MIXED_TRAIN}")
echo "[aot] Filtered: ${FILTERED_TOTAL}/${RAW_TOTAL} samples kept -> ${FILTERED_TRAIN}"

# =========================================================
# Step C+: 答案选项重平衡（binary A/B + 3-way A/B/C）
# 消除离线过滤后因位置偏差导致的答案分布不均
# =========================================================
BALANCED_TRAIN="${DATA_DIR}/mixed_train.offline_filtered.balanced.jsonl"
if [[ ! -f "${BALANCED_TRAIN}" || "${FORCE_FILTER:-false}" == "true" ]]; then
  echo "[aot] Rebalancing answer distribution -> ${BALANCED_TRAIN}"
  python3 "${REPO_ROOT}/proxy_data/temporal_aot/rebalance_aot_answers.py" \
    --input-jsonl  "${FILTERED_TRAIN}" \
    --output-jsonl "${BALANCED_TRAIN}" \
    --problem-types "aot_v2t,aot_t2v,aot_3way_v2t,aot_3way_t2v" \
    --balance-scope problem_type \
    --seed 42
else
  echo "[aot] Reusing balanced file: ${BALANCED_TRAIN} (set FORCE_FILTER=true to redo)"
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
  worker.rollout.gpu_memory_utilization=0.5 \
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
  data.val_batch_size=8
