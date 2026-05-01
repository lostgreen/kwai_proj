#!/usr/bin/env bash
# Base (TG + LLaVA) + Event Logic harder 10k, 8B EMA-GRPO.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_8b.sh"
source "${SCRIPT_DIR}/event_logic_common.sh"

export EXP_NAME="${BASE_LOGIC_8B_EXP_NAME:-composition_base_logic_el10k_mf256_8b_ema}"
export TASKS="${TASKS:-tg mcq event_logic}"
BASE_LOGIC_4B_DATA_DIR="${BASE_LOGIC_4B_DATA_DIR:-${ABLATION_4B_EXPERIMENTS_DIR}/composition_base_logic_el10k_mf256_ema}"
export TRAIN_FILE="${TRAIN_FILE:-${BASE_LOGIC_4B_DATA_DIR}/train.jsonl}"
export TEST_FILE="${TEST_FILE:-${BASE_LOGIC_4B_DATA_DIR}/val.jsonl}"

source "${REPO_ROOT_LOCAL}/local_scripts/run_multi_task.sh"
