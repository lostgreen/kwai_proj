#!/usr/bin/env bash
# =============================================================
# Run R1 F1-IoU / GRPO frame-count ablations.
#
# Usage:
#   EXPS="MF128 MF64" bash local_scripts/hier_seg_ablations/frame_ablation/run_frame_ablation.sh
#   MAX_STEPS=30 EXPS="MF128" bash local_scripts/hier_seg_ablations/frame_ablation/run_frame_ablation.sh  # smoke run
#   MIX_ONLY=true EXPS="MF128 MF64" bash local_scripts/hier_seg_ablations/frame_ablation/run_frame_ablation.sh
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

EXPS="${EXPS:-MF128 MF64}"

echo "[frame_ablation] Starting: experiments=${EXPS}, max_steps=${MAX_STEPS:-<1 epoch>}, MIX_ONLY=${MIX_ONLY:-false}"
echo "[frame_ablation] $(date)"
echo "============================================================"

for EXP in ${EXPS}; do
    echo ""
    echo "------------------------------------------------------------"
    echo "[frame_ablation] Running ${EXP} ..."
    echo "------------------------------------------------------------"

    case "${EXP}" in
        MF128) bash "${SCRIPT_DIR}/exp_r1_f1iou_grpo_full20k_mf128.sh" ;;
        MF64)  bash "${SCRIPT_DIR}/exp_r1_f1iou_grpo_full20k_mf64.sh" ;;
        *)     echo "[frame_ablation] Unknown experiment: ${EXP}" >&2; exit 1 ;;
    esac

    echo "[frame_ablation] Completed ${EXP} at $(date)"
done

echo ""
echo "============================================================"
echo "[frame_ablation] All done."
