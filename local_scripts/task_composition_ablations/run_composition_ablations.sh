#!/usr/bin/env bash
# Run task-composition ablations sequentially.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPS="${EXPS:-BASE_SEG BASE_AOT BASE_SEG_AOT}"

echo "[composition_ablation] experiments=${EXPS}"
echo "[composition_ablation] $(date)"

for EXP in ${EXPS}; do
    case "${EXP}" in
        BASE_SEG)
            bash "${SCRIPT_DIR}/exp_base_seg.sh"
            ;;
        BASE_AOT)
            bash "${SCRIPT_DIR}/exp_base_aot.sh"
            ;;
        BASE_LOGIC)
            bash "${SCRIPT_DIR}/exp_base_logic.sh"
            ;;
        BASE_SEG_AOT)
            bash "${SCRIPT_DIR}/exp_base_seg_aot.sh"
            ;;
        BASE_SEG_LOGIC_AOT)
            bash "${SCRIPT_DIR}/exp_base_seg_logic_aot.sh"
            ;;
        *)
            echo "[composition_ablation] Unknown experiment: ${EXP}" >&2
            exit 1
            ;;
    esac
done
