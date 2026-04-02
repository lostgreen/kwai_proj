#!/usr/bin/env bash
# exp_seg_phase_t2v.sh — T2V 3-way, phase level
# forward 阶段列表 → 从三个 L1 clip 中选匹配的（A/B/C）
# 注：phase_t2v 依赖 phase_v2t 池，因此 SEG_TASKS 包含两者
set -euo pipefail
set -x

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-phase_v2t phase_t2v}"
EXP_NAME="${EXP_NAME:-seg_aot_exp6_phase_t2v}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
