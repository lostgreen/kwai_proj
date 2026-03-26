#!/usr/bin/env bash
# =============================================================
# 实验 6 — All Mixed + AI Filter
#   三种任务全部混合，启用 VLM 因果过滤
#   验证 AI 过滤对数据质量和训练效果的影响
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

EXP_NAME="${EXP_NAME:-el_ablation_exp6_all_filtered}"
DATA_DIR="${DATA_DIR:-${EL_DATA_ROOT}/exp6}"
mkdir -p "${DATA_DIR}"

ADD_PER_VIDEO=2
REPLACE_PER_VIDEO=2
SORT_PER_VIDEO=1

# 启用 AI 因果过滤
FILTER_AI=true

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
