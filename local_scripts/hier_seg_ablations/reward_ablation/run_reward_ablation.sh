#!/usr/bin/env bash
# =============================================================
# run_reward_ablation.sh — Reward 消融实验
#
# 两组实验（全部 L2+L3，固定 V3-prompt V2）:
#   R1: F1-IoU (baseline, 匈牙利匹配)
#   R2: Boundary-Aware (边界命中 F1 + 段数准确性 + 覆盖率 IoU)
#
# 用法:
#   bash run_reward_ablation.sh              # 全部运行
#   EXPS="R1" bash run_reward_ablation.sh    # 仅 R1
#   MAX_STEPS=30 bash run_reward_ablation.sh # 快速调试
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXPS="${EXPS:-R1 R2}"
MAX_STEPS="${MAX_STEPS:-60}"

_START=$(date +%s)
echo "[reward_ablation] Starting: experiments=${EXPS}, max_steps=${MAX_STEPS}"
echo "[reward_ablation] $(date)"
echo "============================================================"

for EXP in ${EXPS}; do
  echo ""
  echo "------------------------------------------------------------"
  echo "[reward_ablation] Running ${EXP} ..."
  echo "------------------------------------------------------------"

  case "${EXP}" in
    R1) MAX_STEPS="${MAX_STEPS}" bash "${SCRIPT_DIR}/exp_r1_f1iou.sh" ;;
    R2) MAX_STEPS="${MAX_STEPS}" bash "${SCRIPT_DIR}/exp_r2_boundary.sh" ;;
    *)  echo "[reward_ablation] Unknown experiment: ${EXP}" >&2; exit 1 ;;
  esac

  echo "[reward_ablation] Completed ${EXP} at $(date)"
done

_END=$(date +%s)
echo ""
echo "============================================================"
echo "[reward_ablation] All done. Elapsed: $(( (_END - _START) / 60 ))m"
