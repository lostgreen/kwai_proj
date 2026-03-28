#!/usr/bin/env bash
# exp_seg_event_v2t.sh — V2T 3-way, event level
# L2 window clip → 哪个事件列表顺序正确?（forward/shuffle/reversed，A/B/C）
set -euo pipefail
set -x

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-event_v2t}"
EXP_NAME="${EXP_NAME:-seg_aot_exp3_event_v2t}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
