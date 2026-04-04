#!/usr/bin/env bash
# exp_t2v_binary.sh — T2V binary MCQ: action-level (文本→视频顺序二选一)
# 数据: youcook2_seg_aot/t2v_binary/
set -euo pipefail
set -x

export MAX_STEPS="${MAX_STEPS:-120}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-seg_aot_action_t2v}"
EXP_NAME="${EXP_NAME:-t2v_binary}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
