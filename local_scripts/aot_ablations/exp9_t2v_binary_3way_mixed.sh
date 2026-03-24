#!/usr/bin/env bash
# =============================================================
# 实验 9 — T2V-Binary+3way Mixed
#   从 exp3 (aot_t2v) 和 exp4 (aot_3way_t2v) 的已过滤数据中
#   各采 500 条，组成 1000 条混合训练集
#
#   消融意义:
#     vs exp3: binary 基础上加入 3way 信号，是否互补
#     vs exp4: 3way 基础上加入 binary 信号，是否互补
#     vs exp7: V2T 混合 vs T2V 混合哪个更有效
#     → 测试 T2V binary 和 3way 混合训练的效果
#
#   前提: exp3 和 exp4 必须已完成离线筛选（report 文件存在）
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- 实验标识 ----
EXP_NAME="${EXP_NAME:-aot_ablation_exp9_t2v_binary_3way_mixed}"

# ---- 数据目录 ----
DATA_DIR="${DATA_DIR:-${AOT_DATA_ROOT}/ablations_refined/exp9}"
mkdir -p "${DATA_DIR}"

# ---- 依赖实验路径 ----
EXP3_DIR="${AOT_DATA_ROOT}/ablations_refined/exp3"
EXP4_DIR="${AOT_DATA_ROOT}/ablations_refined/exp4"

# ---- 跨实验采样配置（CURATE_PER_TYPE_QUOTA 留空 → 自动按源文件 min 数量均衡）----
CURATE_REPORT_JSONLS="${EXP3_DIR}/offline_filter_report.jsonl,${EXP4_DIR}/offline_filter_report.jsonl"
CURATE_TRAIN_JSONLS="${EXP3_DIR}/mixed_train.jsonl,${EXP4_DIR}/mixed_train.jsonl"

source "$(dirname "${BASH_SOURCE[0]}")/launch_train_cross_exp.sh"
