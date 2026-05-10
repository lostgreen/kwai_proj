#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

target="${1:-aot_nocot}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${target}" in
    aot_nocot|run_aot_nocot.sh) target_script="${SCRIPT_DIR}/run_aot_nocot.sh" ;;
    aot_cot|run_aot_cot.sh) target_script="${SCRIPT_DIR}/run_aot_cot.sh" ;;
    seg_nocot|run_seg_nocot.sh) target_script="${SCRIPT_DIR}/run_seg_nocot.sh" ;;
    seg_cot|run_seg_cot.sh) target_script="${SCRIPT_DIR}/run_seg_cot.sh" ;;
    logic_nocot|run_logic_nocot.sh) target_script="${SCRIPT_DIR}/run_logic_nocot.sh" ;;
    logic_cot|run_logic_cot.sh) target_script="${SCRIPT_DIR}/run_logic_cot.sh" ;;
    *)
        echo "Usage: $0 [aot_nocot|aot_cot|seg_nocot|seg_cot|logic_nocot|logic_cot] [extra args]" >&2
        exit 2
        ;;
esac

export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
export TP_SIZE="${TP_SIZE:-1}"
export ROLLOUT_BS="${ROLLOUT_BS:-8}"
export GLOBAL_BS="${GLOBAL_BS:-${ROLLOUT_BS}}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"

exec bash "${target_script}" "$@"
