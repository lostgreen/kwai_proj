#!/usr/bin/env bash
# exp_seg_phase_v2t.sh — V2T 3-way, phase level
# L1 全视频 → 哪个阶段列表顺序正确?（forward / shuffled / reversed，A/B/C）
set -euo pipefail
set -x

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-phase_v2t}"
EXP_NAME="${EXP_NAME:-seg_aot_exp5_phase_v2t}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
