#!/usr/bin/env bash
# Base (TG + LLaVA) + AoT 10k, 8B EMA-GRPO.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_8b.sh"

export EXP_NAME="${BASE_AOT_8B_EXP_NAME:-composition_base_aot_aot10k_mf256_8b_ema}"
export TASKS="${TASKS:-tg mcq aot}"
BASE_AOT_4B_DATA_DIR="${BASE_AOT_4B_DATA_DIR:-${EXPERIMENTS_DIR}/composition_base_aot_aot10k_mf256_ema}"
export TRAIN_FILE="${TRAIN_FILE:-${BASE_AOT_4B_DATA_DIR}/train.jsonl}"
export TEST_FILE="${TEST_FILE:-${BASE_AOT_4B_DATA_DIR}/val.jsonl}"

source "${REPO_ROOT_LOCAL}/local_scripts/run_multi_task.sh"
