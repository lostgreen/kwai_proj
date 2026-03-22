#!/usr/bin/env bash
# =============================================================
# 实验 3 — T2V-Binary
#   AoT 数据: aot_t2v (给 caption 选 composite video, A/B)
#   流程: MCQ 构造 → 混合 temporal_seg → 离线筛选 → 训练
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- 实验标识 ----
EXP_NAME="${EXP_NAME:-aot_ablation_exp3_t2v_binary}"

# ---- 数据目录（放在 AOT_DATA_ROOT 下统一管理）----
DATA_DIR="${DATA_DIR:-${AOT_DATA_ROOT}/ablations_refined/exp3}"
mkdir -p "${DATA_DIR}"

# ---- MCQ 输出路径 ----
V2T_OUTPUT=""
T2V_OUTPUT="${DATA_DIR}/t2v_binary.jsonl"
THREEWAY_V2T_OUTPUT=""
THREEWAY_T2V_OUTPUT=""

# mix_aot_with_youcook2 需要至少一个非空的 aot 输入，这里传 t2v
# 对应 --v2t-jsonl 用 /dev/null 即可跳过
V2T_FOR_MIX=""   # 空=不传入 v2t

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
