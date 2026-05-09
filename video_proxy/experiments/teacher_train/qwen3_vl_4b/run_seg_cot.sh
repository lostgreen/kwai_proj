#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FAMILY="qwen3_vl"
MODEL_SIZE="4b"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
TEACHER_KIND="seg"
SOURCE_EXP_NAME="${SOURCE_EXP_NAME:-composition_base_seg_hier10k_mf256_ema}"
TASKS="${TASKS:-tg mcq hier_seg}"
COT_MODE="${COT_MODE:-true}"

source "${SCRIPT_DIR}/../../../training/recipes/single_teacher_from_experiment.sh"
