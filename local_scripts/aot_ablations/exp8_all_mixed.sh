#!/usr/bin/env bash
# =============================================================
# 实验 8 — All-Mixed（四种任务等量混合）
#   从 exp1-exp4 的已过滤数据中各采 250 条，组成 1000 条训练集:
#     aot_v2t      (250) — from exp1
#     aot_3way_v2t (250) — from exp2
#     aot_t2v      (250) — from exp3
#     aot_3way_t2v (250) — from exp4
#
#   消融意义:
#     vs exp5 (mixed binary): 加入 3way 是否有额外收益
#     vs exp6 (mixed 3way):   加入 binary 是否有额外收益
#     vs exp7 (V2T mixed):    加入 T2V 方向是否有额外收益
#     → 测试最大任务多样性（4种类型等量）的效果
#
#   前提: exp1-exp4 必须已完成离线筛选（report 文件存在）
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- 实验标识 ----
EXP_NAME="${EXP_NAME:-aot_ablation_exp8_all_mixed}"

# ---- 数据目录 ----
DATA_DIR="${DATA_DIR:-${AOT_DATA_ROOT}/ablations_refined/exp8}"
mkdir -p "${DATA_DIR}"

# ---- 依赖实验路径 ----
EXP1_DIR="${AOT_DATA_ROOT}/ablations_refined/exp1"
EXP2_DIR="${AOT_DATA_ROOT}/ablations_refined/exp2"
EXP3_DIR="${AOT_DATA_ROOT}/ablations_refined/exp3"
EXP4_DIR="${AOT_DATA_ROOT}/ablations_refined/exp4"

# ---- 跨实验采样配置 ----
CURATE_REPORT_JSONLS="${EXP1_DIR}/offline_filter_report.jsonl,${EXP2_DIR}/offline_filter_report.jsonl,${EXP3_DIR}/offline_filter_report.jsonl,${EXP4_DIR}/offline_filter_report.jsonl"
CURATE_TRAIN_JSONLS="${EXP1_DIR}/mixed_train.jsonl,${EXP2_DIR}/mixed_train.jsonl,${EXP3_DIR}/mixed_train.jsonl,${EXP4_DIR}/mixed_train.jsonl"
CURATE_PER_TYPE_QUOTA='{"aot_v2t": 250, "aot_3way_v2t": 250, "aot_t2v": 250, "aot_3way_t2v": 250}'

source "$(dirname "${BASH_SOURCE[0]}")/launch_train_cross_exp.sh"
