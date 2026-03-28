#!/usr/bin/env bash
# exp_seg_action_v2t.sh — V2T binary, action level
# L3 event clip → 哪个动作列表顺序正确?（forward vs reversed，A/B）
set -euo pipefail
set -x

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-action_v2t}"
EXP_NAME="${EXP_NAME:-seg_aot_exp1_action_v2t}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
