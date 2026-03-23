#!/usr/bin/env bash
# =============================================================
# 实验 1 — Temporal Grounding (No CoT)
#   纯时间定位任务，直接输出 <events>[[s, e]]</events>
#   数据：TimeRFT 2.5K 筛选后 ≤256s → 2148 条
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- 实验标识 ----
EXP_NAME="${EXP_NAME:-tg_ablation_exp1_no_cot}"

# ---- 数据（无 CoT 版本）----
# 如果数据不存在，先生成
DATA_DIR="${REPO_ROOT}/proxy_data/temporal_grounding/data"
TRAIN_FILE="${DATA_DIR}/timerft_train_max256s_easyr1_clean.jsonl"
TEST_FILE="${DATA_DIR}/tvgbench_val_max256s_easyr1_200_clean.jsonl"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[tg] Generating no_cot data ..."
  python3 "${REPO_ROOT}/proxy_data/temporal_grounding/build_dataset.py" \
    --timerft_json "${REPO_ROOT}/proxy_data/temporal_grounding/annotations/train_2k5.json" \
    --tvgbench_json "${REPO_ROOT}/proxy_data/temporal_grounding/annotations/tvgbench.json" \
    --video_base "${TG_DATA_ROOT}" \
    --output_dir "${DATA_DIR}" \
    --max_duration 256 \
    --mode no_cot
fi

# ---- Response 长度：无 CoT 不需要很长 ----
MAX_RESPONSE_LEN=256

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
