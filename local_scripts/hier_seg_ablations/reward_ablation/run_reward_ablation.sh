#!/usr/bin/env bash
# =============================================================
# run_reward_ablation.sh — Reward 消融实验
#
# 三组实验（全部 L2+L3，固定 V3-prompt V2）:
#   R1: F1-IoU (baseline, 匈牙利匹配)           [0, 1]
#   R3: DP-F1 + Instance Count (DP 顺序匹配)    [0, 2]
#   R4: Segment Matching (全局覆盖 + 局部 NGIoU) [0, 1]
#
# 用法:
#   bash run_reward_ablation.sh              # 全部运行
#   EXPS="R1" bash run_reward_ablation.sh    # 仅 R1
#   EXPS="R3 R4" bash run_reward_ablation.sh # 仅新 reward
#   MAX_STEPS=30 bash run_reward_ablation.sh # 快速调试
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXPS="${EXPS:-R1 R3 R4}"
MAX_STEPS="${MAX_STEPS:-30}"

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
    R3) MAX_STEPS="${MAX_STEPS}" bash "${SCRIPT_DIR}/exp_r3_dp_f1.sh" ;;
    R4) MAX_STEPS="${MAX_STEPS}" bash "${SCRIPT_DIR}/exp_r4_seg_match.sh" ;;
    *)  echo "[reward_ablation] Unknown experiment: ${EXP}" >&2; exit 1 ;;
  esac

  echo "[reward_ablation] Completed ${EXP} at $(date)"
done

_END=$(date +%s)
echo ""
echo "============================================================"
echo "[reward_ablation] All done. Elapsed: $(( (_END - _START) / 60 ))m"
