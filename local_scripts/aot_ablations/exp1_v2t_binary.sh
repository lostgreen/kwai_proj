#!/usr/bin/env bash
# =============================================================
# 实验 1 — V2T-Binary
#   AoT 数据: aot_v2t (forward vs reverse, A/B)
#   流程: MCQ 构造 → 混合 temporal_seg → 离线筛选 → 训练
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- 实验标识 ----
EXP_NAME="${EXP_NAME:-aot_ablation_exp1_v2t_binary}"

# ---- 每个实验独立的数据目录（放在 AOT_DATA_ROOT 下统一管理）----
DATA_DIR="${DATA_DIR:-${AOT_DATA_ROOT}/ablations_refined/exp1}"
mkdir -p "${DATA_DIR}"

# ---- MCQ 输出路径 ----
V2T_OUTPUT="${DATA_DIR}/v2t_binary.jsonl"
T2V_OUTPUT=""
THREEWAY_V2T_OUTPUT=""
THREEWAY_T2V_OUTPUT=""

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
