#!/usr/bin/env bash
# Base (TG + LLaVA) + HierSeg 10k + Event Logic harder 10k + AoT 10k, 4B EMA-GRPO.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${SCRIPT_DIR}/event_logic_common.sh"

export EXP_NAME="${EXP_NAME:-composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema}"
export TASKS="${TASKS:-tg mcq hier_seg event_logic aot}"

source "${REPO_ROOT_LOCAL}/local_scripts/run_multi_task.sh"
