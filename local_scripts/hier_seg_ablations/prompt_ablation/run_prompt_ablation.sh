#!/usr/bin/env bash
# =============================================================
# run_prompt_ablation.sh — Prompt 消融实验
#
# 两组实验（全部 L2+L3，固定 F1-IoU reward）:
#   PA1: 原始标注 prompt (cooking 领域词, 无稀疏约束)
#   PA2: V3 边界判据 prompt (domain-agnostic, sparse-aware, 硬规则)
#
# 用法:
#   bash run_prompt_ablation.sh                # 全部运行
#   EXPS="PA1" bash run_prompt_ablation.sh     # 仅 PA1
#   MAX_STEPS=30 bash run_prompt_ablation.sh   # 快速调试
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXPS="${EXPS:-PA1 PA2}"
MAX_STEPS="${MAX_STEPS:-60}"

_START=$(date +%s)
echo "[prompt_ablation] Starting: experiments=${EXPS}, max_steps=${MAX_STEPS}"
echo "[prompt_ablation] $(date)"
echo "============================================================"

for EXP in ${EXPS}; do
  echo ""
  echo "------------------------------------------------------------"
  echo "[prompt_ablation] Running ${EXP} ..."
  echo "------------------------------------------------------------"

  case "${EXP}" in
    PA1) MAX_STEPS="${MAX_STEPS}" bash "${SCRIPT_DIR}/exp_pa1_original.sh" ;;
    PA2) MAX_STEPS="${MAX_STEPS}" bash "${SCRIPT_DIR}/exp_pa2_v3boundary.sh" ;;
    *)   echo "[prompt_ablation] Unknown experiment: ${EXP}" >&2; exit 1 ;;
  esac

  echo "[prompt_ablation] Completed ${EXP} at $(date)"
done

_END=$(date +%s)
echo ""
echo "============================================================"
echo "[prompt_ablation] All done. Elapsed: $(( (_END - _START) / 60 ))m"
