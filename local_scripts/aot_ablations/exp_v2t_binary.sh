#!/usr/bin/env bash
# exp_v2t_binary.sh — V2T binary MCQ: action-level (视频→文本顺序二选一)
# 数据: youcook2_seg_aot/v2t_binary/
set -euo pipefail
set -x

export MAX_STEPS="${MAX_STEPS:-120}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-seg_aot_action_v2t}"
EXP_NAME="${EXP_NAME:-v2t_binary}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
