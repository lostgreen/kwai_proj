#!/usr/bin/env bash
# =============================================================
# run_v2_ablation.sh — 批量运行 V2 Prompt 消融实验
#
# 顺序运行 V1 → V2 → V3 → V4，每个变体使用相同的层级配置
#
# 用法:
#   # 全三层 V1-V4（默认）
#   bash run_v2_ablation.sh
#
#   # 仅 L2 对比四种变体
#   LEVELS="L2" bash run_v2_ablation.sh
#
#   # 指定 MAX_STEPS 减少训练轮次（调试用）
#   MAX_STEPS=30 LEVELS="L2" bash run_v2_ablation.sh
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

LEVELS="${LEVELS:-L1 L2 L3}"
VARIANTS="${VARIANTS:-V1 V2 V3 V4}"
MAX_STEPS="${MAX_STEPS:-60}"

_START=$(date +%s)
echo "[v2_ablation] Starting batch: variants=${VARIANTS}, levels=${LEVELS}, max_steps=${MAX_STEPS}"
echo "[v2_ablation] $(date)"
echo "============================================================"

for VARIANT in ${VARIANTS}; do
  echo ""
  echo "------------------------------------------------------------"
  echo "[v2_ablation] Running VARIANT=${VARIANT}, LEVELS='${LEVELS}'"
  echo "------------------------------------------------------------"

  MAX_STEPS="${MAX_STEPS}" \
  VARIANT="${VARIANT}" \
  LEVELS="${LEVELS}" \
  bash "${SCRIPT_DIR}/exp_v2_ablation.sh" "${VARIANT}" "${LEVELS}"

  echo "[v2_ablation] Completed VARIANT=${VARIANT} at $(date)"
done

_END=$(date +%s)
echo ""
echo "============================================================"
echo "[v2_ablation] All variants completed. Elapsed: $(( (_END - _START) / 60 ))m"
