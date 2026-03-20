#!/usr/bin/env bash
# =============================================================
# 实验 5 — Mixed-Binary
#   AoT 数据: aot_v2t + aot_t2v（均 A/B）
#   流程: MCQ 构造 → 混合 temporal_seg → 离线筛选 → 训练
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- 实验标识 ----
EXP_NAME="${EXP_NAME:-aot_ablation_exp5_mixed_binary}"

# ---- 数据目录（放在 AOT_DATA_ROOT 下统一管理）----
DATA_DIR="${DATA_DIR:-${AOT_DATA_ROOT}/ablations/exp5}"
mkdir -p "${DATA_DIR}"

# ---- MCQ 输出路径 ----
V2T_OUTPUT="${DATA_DIR}/v2t_binary.jsonl"
T2V_OUTPUT="${DATA_DIR}/t2v_binary.jsonl"
FOURWAY_V2T_OUTPUT=""
FOURWAY_T2V_OUTPUT=""

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
