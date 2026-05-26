#!/usr/bin/env bash
set -euo pipefail

source /home/xuboshen/Anaconda/bin/activate
conda activate one-thinker
export PATH="/home/xuboshen/Anaconda/envs/one-thinker/bin:$PATH"
export http_proxy=http://oversea-squid2.ko.txyun:11080
export https_proxy=http://oversea-squid2.ko.txyun:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

cd /home/xuboshen/zgw/EasyR1

OUTPUT_ROOT=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/results_nextvideo_smoke_2gpu_320
NEXTVIDEO_ROOT=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/NeXTVideo
MODEL_PATH=/m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct
MCQ_JSONL="${OUTPUT_ROOT}/nextvideo_mcq_all.jsonl"
ROLLOUT_OUTPUT="${OUTPUT_ROOT}/rollout_kept.jsonl"
ROLLOUT_REPORT="${OUTPUT_ROOT}/rollout_report.jsonl"
REWARD_FN="/home/xuboshen/zgw/EasyR1/video_proxy/data/base_sources/mcq/rollout/reward.py:compute_score"
ROLLOUT_SCRIPT="/home/xuboshen/zgw/EasyR1/video_proxy/training/tools/offline_rollout_filter.py"

mkdir -p "${OUTPUT_ROOT}"

echo "=== smoke env ==="
date
pwd
git log --oneline -1
python -V
which python
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

echo "=== prepare nextvideo jsonl ==="
python video_proxy/data/base_sources/mcq/nextvideo/prepare_nextvideo.py \
  --input "${NEXTVIDEO_ROOT}/train.jsonl" \
  --output "${MCQ_JSONL}" \
  --dataset-root "${NEXTVIDEO_ROOT}" \
  --split train \
  --verify-videos \
  --summary-json "${OUTPUT_ROOT}/nextvideo_mcq_all_summary.json"

echo "=== rollout smoke: max_samples=320, batch_size=32, num_rollouts=8, num_gpus=2 ==="
for i in 0 1; do
  CUDA_VISIBLE_DEVICES="${i}" \
  VERL_GPU_SIGNAL_PATH="/tmp/nextvideo_smoke_gpu_phase_gpu${i}" \
  python "${ROLLOUT_SCRIPT}" \
    --input_jsonl "${MCQ_JSONL}" \
    --output_jsonl "${OUTPUT_ROOT}/_shard${i}_kept.jsonl" \
    --report_jsonl "${OUTPUT_ROOT}/_shard${i}_report.jsonl" \
    --model_path "${MODEL_PATH}" \
    --reward_function "${REWARD_FN}" \
    --backend vllm \
    --num_rollouts 8 \
    --temperature 0.7 \
    --max_new_tokens 256 \
    --gpu_memory_utilization 0.82 \
    --max_num_batched_tokens 24576 \
    --batch_size 32 \
    --max_samples 320 \
    --min_mean_reward 0.0 \
    --max_mean_reward 1.0 \
    --seed 42 \
    --tensor_parallel_size 1 \
    --shard_id "${i}" \
    --num_shards 2 &
done
wait

cat "${OUTPUT_ROOT}"/_shard*_kept.jsonl > "${ROLLOUT_OUTPUT}"
cat "${OUTPUT_ROOT}"/_shard*_report.jsonl > "${ROLLOUT_REPORT}"

echo "=== smoke counts ==="
wc -l "${MCQ_JSONL}" "${OUTPUT_ROOT}"/_shard*_report.jsonl "${ROLLOUT_REPORT}" "${ROLLOUT_OUTPUT}"
echo "=== report sample ==="
head -n 2 "${ROLLOUT_REPORT}"
echo "=== done ==="
date
