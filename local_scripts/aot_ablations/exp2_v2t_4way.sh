#!/usr/bin/env bash
# =============================================================
# 实验 2 — V2T-4way（纯 4-way，不混 binary）
#   AoT 数据: aot_4way_v2t (A/B/C/D) — 含 forward + shuffle 两种视频
#   vs exp1(binary): 选项数从 2→4，额外增加 shuffle/hard_neg 干扰
#   流程: MCQ 构造 → 离线筛选 → 答案重平衡 → 训练
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- 实验标识 ----
EXP_NAME="${EXP_NAME:-aot_ablation_exp2_v2t_4way}"

# ---- 数据目录（放在 AOT_DATA_ROOT 下统一管理）----
DATA_DIR="${DATA_DIR:-${AOT_DATA_ROOT}/ablations/exp2}"
mkdir -p "${DATA_DIR}"

# ---- MCQ 输出路径（不设 V2T_OUTPUT，只生成 4-way V2T）----
V2T_OUTPUT=""
T2V_OUTPUT=""
FOURWAY_V2T_OUTPUT="${DATA_DIR}/v2t_4way.jsonl"
FOURWAY_T2V_OUTPUT=""

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
