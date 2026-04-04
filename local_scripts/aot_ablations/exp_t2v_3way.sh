#!/usr/bin/env bash
# exp_t2v_3way.sh — T2V 3-way MCQ: phase + event (文本→视频顺序三选一)
# 数据: youcook2_seg_aot/t2v_3way/
set -euo pipefail
set -x

export MAX_STEPS="${MAX_STEPS:-120}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-seg_aot_phase_t2v seg_aot_event_t2v}"
EXP_NAME="${EXP_NAME:-t2v_3way}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
