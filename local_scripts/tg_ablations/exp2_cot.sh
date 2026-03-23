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
EXP_NAME="${EXP_NAME:-tg_ablation_exp2_cot_v2}"

# ---- 数据（CoT 版本）----
DATA_DIR="${REPO_ROOT}/proxy_data/temporal_grounding/data"
TRAIN_FILE="${DATA_DIR}/timerft_train_max256s_cot_easyr1_clean.jsonl"
TEST_FILE="${DATA_DIR}/tvgbench_val_max256s_cot_easyr1_200_clean.jsonl"

# 从已有的 no_cot JSONL 转换（只替换 prompt 模板）
NOCOT_TRAIN="${DATA_DIR}/timerft_train_max256s_easyr1_clean.jsonl"
NOCOT_TEST="${DATA_DIR}/tvgbench_val_max256s_easyr1_200_clean.jsonl"
CONVERT_SCRIPT="${REPO_ROOT}/proxy_data/temporal_grounding/convert_nocot_to_cot.py"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[tg] Converting no_cot -> cot (train) ..."
  python3 "${CONVERT_SCRIPT}" "${NOCOT_TRAIN}" "${TRAIN_FILE}"
fi
if [[ ! -f "${TEST_FILE}" ]]; then
  echo "[tg] Converting no_cot -> cot (val) ..."
  python3 "${CONVERT_SCRIPT}" "${NOCOT_TEST}" "${TEST_FILE}"
fi

# ---- CoT 需要更长的 response 空间 ----
MAX_RESPONSE_LEN=1024

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
