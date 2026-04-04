#!/usr/bin/env bash
# exp_v2t_3way.sh — V2T 3-way MCQ: phase + event (视频→文本顺序三选一)
# 数据: youcook2_seg_aot/v2t_3way/
set -euo pipefail
set -x

export MAX_STEPS="${MAX_STEPS:-120}"

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-seg_aot_phase_v2t seg_aot_event_v2t}"
EXP_NAME="${EXP_NAME:-v2t_3way}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
