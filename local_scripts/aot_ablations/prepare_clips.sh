#!/usr/bin/env bash
# =============================================================
# prepare_clips.sh — 独立切分 L1/L2/L3 原子视频 clips
#
# 三条 proxy 流水线（temporal_aot、event_logic、hier_seg）共用同一批 clips。
# 在训练前单独执行，确保数据准备和训练解耦。
#
# 输出结构:
#   {CLIP_ROOT}/
#   ├── L1/  {key}_L1_ph{id}_{start}_{end}.mp4   (phase clips, 1fps)
#   ├── L2/  {key}_L2_ev{id}_{start}_{end}.mp4   (event clips, 2fps)
#   └── L3/  {key}_L3_act{id}_ev{parent}_{start}_{end}.mp4 (action clips, 2fps)
#
# 用法:
#   bash local_scripts/aot_ablations/prepare_clips.sh
#   LEVELS="L1 L2" bash local_scripts/aot_ablations/prepare_clips.sh
#   bash local_scripts/aot_ablations/prepare_clips.sh --dry-run
# =============================================================
set -euo pipefail
set -x

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# ---- 可覆盖参数 ----
LEVELS="${LEVELS:-L1 L2 L3}"
L1_FPS="${L1_FPS:-1}"
L2L3_FPS="${L2L3_FPS:-2}"
WORKERS="${WORKERS:-8}"

echo "[prepare-clips] Annotation dir  : ${ANNOTATION_DIR}"
echo "[prepare-clips] Output dir      : ${CLIP_ROOT}"
echo "[prepare-clips] Levels          : ${LEVELS}"
echo "[prepare-clips] L1 FPS          : ${L1_FPS}"
echo "[prepare-clips] L2/L3 FPS       : ${L2L3_FPS}"
echo "[prepare-clips] Workers         : ${WORKERS}"

CMD=(
  python3 "${REPO_ROOT}/proxy_data/youcook2_seg/prepare_all_clips.py"
  --annotation-dir "${ANNOTATION_DIR}"
  --output-dir "${CLIP_ROOT}"
  --levels ${LEVELS}
  --l1-fps "${L1_FPS}"
  --l2l3-fps "${L2L3_FPS}"
  --workers "${WORKERS}"
  --min-phases "${MIN_PHASES}"
  --min-events "${MIN_EVENTS}"
  --min-actions "${MIN_ACTIONS}"
  --complete-only
)

# 如果设置了 SOURCE_VIDEO_DIR，添加参数
if [[ -n "${SOURCE_VIDEO_DIR:-}" ]]; then
  CMD+=(--source-video-dir "${SOURCE_VIDEO_DIR}")
fi

# 追加额外参数（如 --dry-run, --overwrite）
"${CMD[@]}" "$@"

echo "[prepare-clips] Done."
