#!/usr/bin/env bash
# =============================================================
# 实验 7 — V2T-Binary+3way Mixed
#   从 exp1 (aot_v2t) 和 exp2 (aot_3way_v2t) 的已过滤数据中
#   各采 500 条，组成 1000 条混合训练集
#
#   消融意义:
#     vs exp1: binary 基础上加入 3way 信号，是否互补
#     vs exp2: 3way 基础上加入 binary 信号，是否互补
#     → 测试 V2T binary 和 3way 混合训练的效果
#
#   前提: exp1 和 exp2 必须已完成离线筛选（report 文件存在）
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- 实验标识 ----
EXP_NAME="${EXP_NAME:-aot_ablation_exp7_v2t_binary_3way_mixed}"

# ---- 数据目录 ----
DATA_DIR="${DATA_DIR:-${AOT_DATA_ROOT}/ablations_refined/exp7}"
mkdir -p "${DATA_DIR}"

# ---- 依赖实验路径 ----
EXP1_DIR="${AOT_DATA_ROOT}/ablations_refined/exp1"
EXP2_DIR="${AOT_DATA_ROOT}/ablations_refined/exp2"

# ---- 跨实验采样配置 ----
CURATE_REPORT_JSONLS="${EXP1_DIR}/offline_filter_report.jsonl,${EXP2_DIR}/offline_filter_report.jsonl"
CURATE_TRAIN_JSONLS="${EXP1_DIR}/mixed_train.jsonl,${EXP2_DIR}/mixed_train.jsonl"
CURATE_PER_TYPE_QUOTA='{"aot_v2t": 500, "aot_3way_v2t": 500}'

source "$(dirname "${BASH_SOURCE[0]}")/launch_train_cross_exp.sh"
