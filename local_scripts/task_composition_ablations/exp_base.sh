#!/usr/bin/env bash
# Base only (TG + LLaVA), 4B EMA-GRPO.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export EXP_NAME="${BASE_EXP_NAME:-composition_base_mf256_ema}"
export TASKS="${TASKS:-tg mcq}"

source "${REPO_ROOT_LOCAL}/local_scripts/run_multi_task.sh"
