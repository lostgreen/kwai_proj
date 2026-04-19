#!/usr/bin/env bash
# =============================================================
# run_el_ablation.sh — Event Logic 三种 proxy 消融
#
# 三组实验 (每组混入 TG + MCQ 基础数据):
#   PN:   Predict Next (MCQ)
#   FB:   Fill Blank (MCQ)
#   SORT: Sequence Sort (jigsaw displacement)
#
# 用法:
#   bash local_scripts/event_logic_ablations/run_el_ablation.sh         # 全部运行
#   EXPS="PN" bash local_scripts/event_logic_ablations/run_el_ablation.sh    # 仅 PN
#   EXPS="FB SORT" bash local_scripts/event_logic_ablations/run_el_ablation.sh
#   MAX_STEPS=30 bash local_scripts/event_logic_ablations/run_el_ablation.sh  # 快速调试
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXPS="${EXPS:-PN FB SORT}"
MAX_STEPS="${MAX_STEPS:-}"

_START=$(date +%s)
echo "[el_ablation] Starting: experiments=${EXPS}${MAX_STEPS:+, max_steps=${MAX_STEPS}}"
echo "[el_ablation] $(date)"
echo "============================================================"

for EXP in ${EXPS}; do
  echo ""
  echo "------------------------------------------------------------"
  echo "[el_ablation] Running ${EXP} ..."
  echo "------------------------------------------------------------"

  case "${EXP}" in
    PN)   ${MAX_STEPS:+MAX_STEPS="${MAX_STEPS}"} bash "${SCRIPT_DIR}/exp_pn.sh" ;;
    FB)   ${MAX_STEPS:+MAX_STEPS="${MAX_STEPS}"} bash "${SCRIPT_DIR}/exp_fb.sh" ;;
    SORT) ${MAX_STEPS:+MAX_STEPS="${MAX_STEPS}"} bash "${SCRIPT_DIR}/exp_sort.sh" ;;
    *)    echo "[el_ablation] Unknown experiment: ${EXP}" >&2; exit 1 ;;
  esac

  echo "[el_ablation] Completed ${EXP} at $(date)"
done

_END=$(date +%s)
echo ""
echo "============================================================"
echo "[el_ablation] All done. Elapsed: $(( (_END - _START) / 60 ))m"
