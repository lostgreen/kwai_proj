#!/usr/bin/env bash
# Run AoT no-CoT offline rollout on 8 GPUs without touching the existing filler.
#
# This script launches one Qwen3-VL-8B vLLM process per GPU (data parallel),
# rewrites shard-local report indices back to global indices, then filters
# hard-but-solvable cases.
#
# Required env:
#   AOT_ROOT      Default: /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot_hardqa_v1
#   MODEL_PATH    Default: /m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct
#
# Optional env:
#   INPUT_JSONL, ROLLOUT_DIR, NUM_GPUS, NUM_ROLLOUTS, MAX_NEW_TOKENS,
#   BATCH_SIZE, GPU_MEM_UTIL, MAX_BATCHED_TOKENS, TARGET_TOTAL, FORCE

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

AOT_ROOT="${AOT_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot_hardqa_v1}"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct}"
INPUT_JSONL="${INPUT_JSONL:-${AOT_ROOT}/merged_raw_nocot/train.jsonl}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${AOT_ROOT}/rollout_nocot}"

NUM_GPUS="${NUM_GPUS:-8}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.82}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-24576}"
MAX_FRAMES="${MAX_FRAMES:-256}"
MAX_PIXELS="${MAX_PIXELS:-49152}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
VIDEO_FPS="${VIDEO_FPS:-2.0}"
TARGET_TOTAL="${TARGET_TOTAL:-5000}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"

REWARD_FUNCTION="${REWARD_FUNCTION:-${REPO_ROOT}/verl/reward_function/mixed_proxy_reward.py:compute_score}"
ROLLOUT_SCRIPT="${REPO_ROOT}/local_scripts/offline_rollout_filter.py"
FILTER_SCRIPT="${REPO_ROOT}/proxy_data/youcook2_seg/temporal_aot/filter_rollout_hard_cases.py"

if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "[aot-rollout] input not found: ${INPUT_JSONL}" >&2
  echo "[aot-rollout] expected no-CoT merged data. Build it first or set INPUT_JSONL." >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[aot-rollout] model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${ROLLOUT_DIR}"

SOURCE_COUNT="$(wc -l < "${INPUT_JSONL}" | tr -d ' ')"
echo "============================================================"
echo " AoT no-CoT rollout"
echo "  input      : ${INPUT_JSONL} (${SOURCE_COUNT} records)"
echo "  output dir : ${ROLLOUT_DIR}"
echo "  model      : ${MODEL_PATH}"
echo "  gpus       : ${NUM_GPUS} data-parallel shards, TP=1"
echo "  rollouts   : ${NUM_ROLLOUTS}"
echo "  batch      : ${BATCH_SIZE}"
echo "  max tokens : ${MAX_NEW_TOKENS}"
echo "  filler     : untouched; phase signals are isolated under /tmp/aot_rollout_unused_phase_gpu*"
echo "============================================================"

REPORT_DONE=0
if [[ "${FORCE}" != "1" && -s "${ROLLOUT_DIR}/rollout_report.jsonl" ]]; then
  REPORT_COUNT="$(wc -l < "${ROLLOUT_DIR}/rollout_report.jsonl" | tr -d ' ')"
  if [[ "${REPORT_COUNT}" -ge "${SOURCE_COUNT}" ]]; then
    echo "[aot-rollout] rollout report already complete (${REPORT_COUNT}/${SOURCE_COUNT}); skipping rollout."
    REPORT_DONE=1
  fi
fi

if [[ "${REPORT_DONE}" != "1" ]]; then
  rm -f \
    "${ROLLOUT_DIR}"/_shard*_kept.jsonl \
    "${ROLLOUT_DIR}"/_shard*_report.jsonl \
    "${ROLLOUT_DIR}"/_shard*_report_global.jsonl \
    "${ROLLOUT_DIR}"/_shard*.log \
    "${ROLLOUT_DIR}/rollout_output.jsonl" \
    "${ROLLOUT_DIR}/rollout_report.jsonl"

  for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "[aot-rollout] launch shard ${i}/${NUM_GPUS} on local GPU ${i}"
    CUDA_VISIBLE_DEVICES="${i}" \
    VERL_GPU_SIGNAL_PATH="/tmp/aot_rollout_unused_phase_gpu${i}" \
    python "${ROLLOUT_SCRIPT}" \
      --input_jsonl "${INPUT_JSONL}" \
      --output_jsonl "${ROLLOUT_DIR}/_shard${i}_kept.jsonl" \
      --report_jsonl "${ROLLOUT_DIR}/_shard${i}_report.jsonl" \
      --model_path "${MODEL_PATH}" \
      --reward_function "${REWARD_FUNCTION}" \
      --backend vllm \
      --num_rollouts "${NUM_ROLLOUTS}" \
      --temperature 0.7 \
      --top_p 0.9 \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --video_fps "${VIDEO_FPS}" \
      --max_frames "${MAX_FRAMES}" \
      --max_pixels "${MAX_PIXELS}" \
      --min_pixels "${MIN_PIXELS}" \
      --tensor_parallel_size 1 \
      --gpu_memory_utilization "${GPU_MEM_UTIL}" \
      --max_num_batched_tokens "${MAX_BATCHED_TOKENS}" \
      --batch_size "${BATCH_SIZE}" \
      --dtype bfloat16 \
      --min_mean_reward 0.0 \
      --max_mean_reward 1.0 \
      --shard_id "${i}" \
      --num_shards "${NUM_GPUS}" \
      --seed "${SEED}" \
      > "${ROLLOUT_DIR}/_shard${i}.log" 2>&1 &
  done

  wait

  python - "${ROLLOUT_DIR}" "${NUM_GPUS}" <<'PY'
import json
import sys
from pathlib import Path

rollout_dir = Path(sys.argv[1])
num_gpus = int(sys.argv[2])
for shard_id in range(num_gpus):
    src = rollout_dir / f"_shard{shard_id}_report.jsonl"
    dst = rollout_dir / f"_shard{shard_id}_report_global.jsonl"
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            local_idx = row.get("index")
            if isinstance(local_idx, int) and local_idx >= 0:
                row["local_index"] = local_idx
                row["shard_id"] = shard_id
                row["num_shards"] = num_gpus
                row["index"] = shard_id + local_idx * num_gpus
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

  cat "${ROLLOUT_DIR}"/_shard*_kept.jsonl > "${ROLLOUT_DIR}/rollout_output.jsonl"
  cat "${ROLLOUT_DIR}"/_shard*_report_global.jsonl > "${ROLLOUT_DIR}/rollout_report.jsonl"
fi

python "${FILTER_SCRIPT}" \
  --report "${ROLLOUT_DIR}/rollout_report.jsonl" \
  --input "${INPUT_JSONL}" \
  --output "${ROLLOUT_DIR}/hard_cases.jsonl" \
  --stats-output "${ROLLOUT_DIR}/hard_cases.stats.json" \
  --min-mean-reward 0.125 \
  --max-mean-reward 0.625 \
  --min-success-count 1 \
  --success-threshold 1.0 \
  --target-total "${TARGET_TOTAL}" \
  --nested-balance-key domain_l1 \
  --seed "${SEED}"

wc -l \
  "${ROLLOUT_DIR}/rollout_output.jsonl" \
  "${ROLLOUT_DIR}/rollout_report.jsonl" \
  "${ROLLOUT_DIR}/hard_cases.jsonl"

python -m json.tool "${ROLLOUT_DIR}/hard_cases.stats.json" | head -120
