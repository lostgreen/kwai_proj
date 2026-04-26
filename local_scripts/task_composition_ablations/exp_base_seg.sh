#!/usr/bin/env bash
# Base (TG + LLaVA) + HierSeg 10k, 4B EMA-GRPO.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export EXP_NAME="${BASE_SEG_EXP_NAME:-composition_base_seg_hier10k_mf256_ema}"
export TASKS="${TASKS:-tg mcq hier_seg}"

source "${REPO_ROOT_LOCAL}/local_scripts/run_multi_task.sh"
