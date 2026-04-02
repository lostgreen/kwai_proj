#!/usr/bin/env bash
# exp_v2t.sh — V2T ablation: 给视频，从 3 种文本顺序中选正确的
# L1 phase + L2 event + L3 action, 比例 1:2:2, 总计 1k
set -euo pipefail
set -x

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-phase_v2t event_v2t action_v2t}"
EXP_NAME="${EXP_NAME:-seg_aot_v2t}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
