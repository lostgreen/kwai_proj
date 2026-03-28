#!/usr/bin/env bash
# exp_seg_action_t2v.sh — T2V binary, action level
# forward 动作列表 → 从两个 L3 clip 中选匹配的（A/B）
set -euo pipefail
set -x

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-action_t2v}"
EXP_NAME="${EXP_NAME:-seg_aot_exp2_action_t2v}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
