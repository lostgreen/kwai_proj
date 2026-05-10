#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FAMILY="qwen2_5_vl"
MODEL_SIZE="7b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen2.5-VL-7B-Instruct}"
TEACHER_KIND="aot"
SOURCE_EXP_NAME="${SOURCE_EXP_NAME:-composition_base_aot_aot10k_mf256_ema}"
TASKS="${TASKS:-tg mcq aot}"
COT_MODE="${COT_MODE:-false}"

source "${SCRIPT_DIR}/../../../training/recipes/single_teacher_from_experiment.sh"
