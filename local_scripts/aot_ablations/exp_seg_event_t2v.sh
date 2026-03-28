#!/usr/bin/env bash
# exp_seg_event_t2v.sh — T2V 3-way, event level
# forward 事件列表 → 从三个 L2 clip 中选匹配的（A/B/C）
# 注: event_t2v 依赖先在全局构建 event_v2t 池，自动包含 event_v2t 数据构建
set -euo pipefail
set -x

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SEG_TASKS="${SEG_TASKS:-event_v2t event_t2v}"
EXP_NAME="${EXP_NAME:-seg_aot_exp4_event_t2v}"

source "$(dirname "${BASH_SOURCE[0]}")/launch_seg_train.sh"
