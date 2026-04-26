#!/usr/bin/env bash
# Offline rollout filtering for Event Logic MCQ tasks.
#
# Usage:
#   INPUT_JSONL=/path/to/train_predict_next.jsonl \
#   MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
#   bash proxy_data/youcook2_seg/event_logic/run_event_logic_rollout.sh
#
# Defaults keep only hard cases whose mean reward is in [0.0, 0.5],
# which removes samples the current policy can answer 100% correctly.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

INPUT_JSONL="${INPUT_JSONL:-}"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
NUM_GPUS="${NUM_GPUS:-1}"
TP_SIZE="${TP_SIZE:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
HARD_MIN_REWARD="${HARD_MIN_REWARD:-0.0}"
HARD_MAX_REWARD="${HARD_MAX_REWARD:-0.5}"
TARGET_TOTAL="${TARGET_TOTAL:-0}"
SEED="${SEED:-42}"

if [[ -z "${INPUT_JSONL}" ]]; then
  echo "Please set INPUT_JSONL=/path/to/event_logic_train.jsonl" >&2
  exit 1
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  INPUT_DIR="$(cd -- "$(dirname -- "${INPUT_JSONL}")" && pwd)"
  OUTPUT_ROOT="${INPUT_DIR}/rollout_results"
fi

mkdir -p "${OUTPUT_ROOT}"

ROLLOUT_SCRIPT="${REPO_ROOT}/local_scripts/offline_rollout_filter.py"
FILTER_SCRIPT="${REPO_ROOT}/proxy_data/youcook2_seg/temporal_aot/filter_rollout_hard_cases.py"
REWARD_FN="${REPO_ROOT}/verl/reward_function/mixed_proxy_reward.py:compute_score"

ROLLOUT_OUTPUT="${OUTPUT_ROOT}/rollout_diverse_unused.jsonl"
ROLLOUT_REPORT="${OUTPUT_ROOT}/rollout_report.jsonl"
HARD_OUTPUT="${OUTPUT_ROOT}/hard_cases.jsonl"

echo "============================================="
echo " Event Logic Hard-Case Rollout"
echo " Input:       ${INPUT_JSONL}"
echo " Model:       ${MODEL_PATH}"
echo " Output:      ${OUTPUT_ROOT}"
echo " GPUs:        ${NUM_GPUS} (TP=${TP_SIZE})"
echo " Rollouts:    ${NUM_ROLLOUTS}"
echo " Hard range:  [${HARD_MIN_REWARD}, ${HARD_MAX_REWARD}]"
echo " Target hard: ${TARGET_TOTAL} (0 = keep all hard cases)"
echo "============================================="

ROLLOUT_COMMON=(
  --input_jsonl "${INPUT_JSONL}"
  --output_jsonl "${ROLLOUT_OUTPUT}"
  --report_jsonl "${ROLLOUT_REPORT}"
  --model_path "${MODEL_PATH}"
  --reward_function "${REWARD_FN}"
  --backend vllm
  --num_rollouts "${NUM_ROLLOUTS}"
  --temperature 1.0
  --top_p 0.9
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --gpu_memory_utilization "${GPU_MEM_UTIL}"
  --max_num_batched_tokens "${MAX_BATCHED_TOKENS}"
  --min_mean_reward 0.0
  --max_mean_reward 1.0
  --seed "${SEED}"
)

if [[ "${TP_SIZE}" -gt 1 ]]; then
  echo "[rollout] Tensor-parallel mode"
  python3 "${ROLLOUT_SCRIPT}" "${ROLLOUT_COMMON[@]}" --tensor_parallel_size "${TP_SIZE}"
elif [[ "${NUM_GPUS}" -gt 1 ]]; then
  echo "[rollout] Data-parallel mode (${NUM_GPUS} GPUs)"
  for i in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i python3 "${ROLLOUT_SCRIPT}" \
      "${ROLLOUT_COMMON[@]}" \
      --tensor_parallel_size 1 \
      --shard_id "$i" --num_shards "${NUM_GPUS}" \
      --output_jsonl "${OUTPUT_ROOT}/_shard${i}_kept.jsonl" \
      --report_jsonl "${OUTPUT_ROOT}/_shard${i}_report.jsonl" &
  done
  wait
  cat "${OUTPUT_ROOT}"/_shard*_kept.jsonl > "${ROLLOUT_OUTPUT}"
  cat "${OUTPUT_ROOT}"/_shard*_report.jsonl > "${ROLLOUT_REPORT}"
  rm -f "${OUTPUT_ROOT}"/_shard*_kept.jsonl "${OUTPUT_ROOT}"/_shard*_report.jsonl
else
  echo "[rollout] Single-GPU mode"
  python3 "${ROLLOUT_SCRIPT}" "${ROLLOUT_COMMON[@]}" --tensor_parallel_size 1
fi

python3 "${FILTER_SCRIPT}" \
  --report "${ROLLOUT_REPORT}" \
  --input "${INPUT_JSONL}" \
  --output "${HARD_OUTPUT}" \
  --min-mean-reward "${HARD_MIN_REWARD}" \
  --max-mean-reward "${HARD_MAX_REWARD}" \
  --target-total "${TARGET_TOTAL}" \
  --seed "${SEED}"

echo "Hard cases written to: ${HARD_OUTPUT}"
