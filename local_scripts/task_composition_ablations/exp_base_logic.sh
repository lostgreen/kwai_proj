#!/usr/bin/env bash
# Base (TG + LLaVA) + Event Logic harder 10k, 4B EMA-GRPO.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${SCRIPT_DIR}/event_logic_common.sh"

export EXP_NAME="${EXP_NAME:-composition_base_logic_el10k_mf256_ema}"
export TASKS="${TASKS:-tg mcq event_logic}"

source "${REPO_ROOT_LOCAL}/local_scripts/run_multi_task.sh"
