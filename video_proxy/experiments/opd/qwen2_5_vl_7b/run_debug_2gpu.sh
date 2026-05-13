#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

target="${1:-3teachers}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${target}" in
    3teachers|run_mopd_3teachers.sh) target_script="${SCRIPT_DIR}/run_mopd_3teachers.sh" ;;
    2teachers|run_mopd_2teachers.sh) target_script="${SCRIPT_DIR}/run_mopd_2teachers.sh" ;;
    *)
        echo "Usage: $0 [3teachers|2teachers] [extra args]" >&2
        exit 2
        ;;
esac

export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
export TP_SIZE="${TP_SIZE:-2}"
export ROLLOUT_BS="${ROLLOUT_BS:-8}"
export GLOBAL_BS="${GLOBAL_BS:-${ROLLOUT_BS}}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export VAL_FREQ="${VAL_FREQ:-10}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export ENABLE_GPU_FILLER="${ENABLE_GPU_FILLER:-false}"
export POST_TRAIN_OCCUPANCY="${POST_TRAIN_OCCUPANCY:-false}"

exec bash "${target_script}" "$@"
