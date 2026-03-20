#!/usr/bin/env bash
# =============================================================
# 实验 6 — Mixed-4way（纯 4-way，V2T+T2V 联合训练）
#   AoT 数据: aot_4way_v2t + aot_4way_t2v（不混 binary）
#   vs exp5(mixed binary): V2T+T2V 联合训练时用 4-way 替代 binary
#   流程: MCQ 构造 → 离线筛选 → 答案重平衡 → 训练
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- 实验标识 ----
EXP_NAME="${EXP_NAME:-aot_ablation_exp6_mixed_4way}"

# ---- 数据目录（放在 AOT_DATA_ROOT 下统一管理）----
DATA_DIR="${DATA_DIR:-${AOT_DATA_ROOT}/ablations/exp6}"
mkdir -p "${DATA_DIR}"

# ---- MCQ 输出路径（不设 binary 输出，只生成 4-way V2T + 4-way T2V）----
V2T_OUTPUT=""
T2V_OUTPUT=""
FOURWAY_V2T_OUTPUT="${DATA_DIR}/v2t_4way.jsonl"
FOURWAY_T2V_OUTPUT="${DATA_DIR}/t2v_4way.jsonl"

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
