#!/usr/bin/env bash
# =============================================================
# 实验 5 — All Mixed (add + replace + sort)
#   三种任务全部混合，无 AI 过滤
#   验证完整 Event Logic 任务集效果
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-el_ablation_exp5_all_mixed}"
DATA_DIR="${DATA_DIR:-${EL_DATA_ROOT}/exp5}"
mkdir -p "${DATA_DIR}"

ADD_PER_VIDEO=2
REPLACE_PER_VIDEO=2
SORT_PER_VIDEO=1

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
