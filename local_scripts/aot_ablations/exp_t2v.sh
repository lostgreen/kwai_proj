#!/usr/bin/env bash
# exp_t2v.sh — T2V ablation: 给文本，从 3 个拼接视频中选匹配的
# L1 phase + L2 event + L3 action, 比例 1:2:2, 总计 1k
set -euo pipefail
set -x

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-phase_t2v event_t2v action_t2v}"
EXP_NAME="${EXP_NAME:-seg_aot_t2v}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
