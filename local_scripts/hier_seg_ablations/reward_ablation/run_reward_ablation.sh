#!/usr/bin/env bash
# =============================================================
# run_reward_ablation.sh — Reward 消融实验入口
#
# 当前仅保留两组有效实验:
#   R1: F1-IoU / Hungarian matching
#   R4: Segment Matching
#
# 两组实验默认都:
#   - 使用 full hier-seg train (HIER_TARGET=0)
#   - 对齐 VideoSSR-like 关键超参
#   - 保留 online filtering=true
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXPS="${EXPS:-R1 R4}"
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
    R4) MAX_STEPS="${MAX_STEPS}" bash "${SCRIPT_DIR}/exp_r4_seg_match.sh" ;;
    *)  echo "[reward_ablation] Unknown experiment: ${EXP}" >&2; exit 1 ;;
  esac

  echo "[reward_ablation] Completed ${EXP} at $(date)"
done

_END=$(date +%s)
echo ""
echo "============================================================"
echo "[reward_ablation] All done. Elapsed: $(( (_END - _START) / 60 ))m"
