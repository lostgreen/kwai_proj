#!/usr/bin/env bash
# =============================================================
# 实验 7 — V2T-3way-FwdOnly（仅 forward_video 样本）
#   AoT 数据: aot_3way_v2t，但只保留 video_direction=="forward" 的条目
#   选项: A/B/C = {forward, reverse, shuffle} caption（随机排列）
#   正确答案: 始终 = forward_caption 所在字母
#
#   消融意义:
#     vs exp1 (binary V2T):  同样只展示 forward_video，但选项数从 2 增到 3，
#                            干扰项从 {reverse} 扩展为 {reverse, shuffle}
#     vs exp2 (3-way V2T full): 去掉 shuffle_video 样本，隔离"识别杂乱视频"的信号，
#                               只测试"从更多干扰中识别正放方向"的难度提升效果
#
#   三组对比: exp1 → exp7 → exp2
#     exp1: binary (2选1, forward/reverse video)
#     exp7: 3-way fwd-only (3选1, 只展示 forward video, harder distractors)
#     exp2: 3-way full (3选1, forward + shuffle video 均训练)
# =============================================================
set -euo pipefail
set -x

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- 实验标识 ----
EXP_NAME="${EXP_NAME:-aot_ablation_exp7_v2t_3way_fwdonly}"

# ---- 数据目录 ----
DATA_DIR="${DATA_DIR:-${AOT_DATA_ROOT}/ablations/exp7}"
mkdir -p "${DATA_DIR}"

# ---- MCQ 输出路径 ----
# 先构造完整 3-way V2T（含 forward + shuffle 样本），之后由 launch_train.sh 过滤
V2T_OUTPUT=""
T2V_OUTPUT=""
THREEWAY_V2T_OUTPUT="${DATA_DIR}/v2t_3way.jsonl"
THREEWAY_T2V_OUTPUT=""

# 告知 launch_train.sh 只保留 forward_video 样本
THREEWAY_V2T_FWD_ONLY=true

source "$(dirname "${BASH_SOURCE[0]}")/launch_train.sh"
