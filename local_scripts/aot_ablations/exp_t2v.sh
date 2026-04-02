#!/usr/bin/env bash
# exp_t2v.sh — T2V ablation: 给文本，从 3 个拼接视频中选匹配的
# L2 event + L3 action (L1 phase 帧预算超标，仅在 V2T 使用)
set -euo pipefail
set -x

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-event_t2v action_t2v}"
EXP_NAME="${EXP_NAME:-seg_aot_t2v}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
