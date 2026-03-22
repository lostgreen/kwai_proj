#!/usr/bin/env bash
# =============================================================
# 实验 2 — Temporal Grounding (CoT)
#   带 Chain-of-Thought 的时间定位，模型先在 <think> 中分析
#   再输出 <events>[[s, e]]</events>
#   数据：TimeRFT 2.5K 筛选后 ≤256s → 2148 条
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- 实验标识 ----
EXP_NAME="${EXP_NAME:-tg_ablation_exp2_cot}"

# ---- 数据（CoT 版本）----
DATA_DIR="${REPO_ROOT}/proxy_data/temporal_grounding/data"
TRAIN_FILE="${DATA_DIR}/timerft_train_max256s_cot_easyr1.jsonl"
TEST_FILE="${DATA_DIR}/tvgbench_val_max256s_cot_easyr1.jsonl"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[tg] Generating cot data ..."
  python3 "${REPO_ROOT}/proxy_data/temporal_grounding/build_dataset.py" \
    --timerft_json "${REPO_ROOT}/proxy_data/temporal_grounding/annotations/train_2k5.json" \
    --tvgbench_json "${REPO_ROOT}/proxy_data/temporal_grounding/annotations/tvgbench.json" \
    --video_base "${TG_DATA_ROOT}" \
    --output_dir "${DATA_DIR}" \
    --max_duration 256 \
    --mode cot
fi

# ---- CoT 需要更长的 response 空间 ----
MAX_RESPONSE_LEN=1024

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
