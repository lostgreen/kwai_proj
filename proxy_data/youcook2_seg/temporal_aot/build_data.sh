#!/usr/bin/env bash
# =============================================================
# build_data.sh — 构造 Seg-AOT V2T/T2V 消融数据
#
# 流程: 切原子 clips → ffmpeg 拼接 → 导出 JSONL
#
# 用法:
#   # 全流程 (切 clips + 构造 V2T/T2V)
#   bash proxy_data/youcook2_seg/temporal_aot/build_data.sh
#
#   # 跳过切 clips (已切好)
#   SKIP_CLIPS=true bash proxy_data/youcook2_seg/temporal_aot/build_data.sh
#
#   # 自定义参数
#   TRAIN_TOTAL=500 LEVEL_RATIO=1:1:1 MAX_PHASES=5 \
#     bash proxy_data/youcook2_seg/temporal_aot/build_data.sh
# =============================================================
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# ---- 路径 ----
HIER_SEG_ROOT="${HIER_SEG_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation}"
ANNOTATION_DIR="${ANNOTATION_DIR:-${HIER_SEG_ROOT}/annotations_checked}"
CLIP_ROOT="${CLIP_ROOT:-${HIER_SEG_ROOT}/clips}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_seg_aot}"
SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-}"

# ---- 过滤参数 ----
MIN_PHASES="${MIN_PHASES:-3}"
MAX_PHASES="${MAX_PHASES:-6}"
MIN_EVENTS="${MIN_EVENTS:-3}"
MAX_EVENTS="${MAX_EVENTS:-8}"
MIN_ACTIONS="${MIN_ACTIONS:-3}"
MAX_ACTIONS="${MAX_ACTIONS:-10}"

# ---- 采样参数 ----
TRAIN_TOTAL="${TRAIN_TOTAL:-1000}"
LEVEL_RATIO="${LEVEL_RATIO:-1:2:2}"
TOTAL_VAL="${TOTAL_VAL:-200}"
SEED="${SEED:-42}"

# ---- Clip 切分参数 ----
L1_FPS="${L1_FPS:-1}"
L2L3_FPS="${L2L3_FPS:-2}"
WORKERS="${WORKERS:-8}"
CONCAT_WORKERS="${CONCAT_WORKERS:-8}"

SKIP_CLIPS="${SKIP_CLIPS:-false}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"

echo "============================================================"
echo "Seg-AOT Data Builder"
echo "  ANNOTATION_DIR : ${ANNOTATION_DIR}"
echo "  CLIP_ROOT      : ${CLIP_ROOT}"
echo "  OUTPUT_ROOT    : ${OUTPUT_ROOT}"
echo "  TRAIN_TOTAL    : ${TRAIN_TOTAL}"
echo "  LEVEL_RATIO    : ${LEVEL_RATIO} (L1:L2:L3)"
echo "  FILTER         : phases=[${MIN_PHASES},${MAX_PHASES}] events=[${MIN_EVENTS},${MAX_EVENTS}] actions=[${MIN_ACTIONS},${MAX_ACTIONS}]"
echo "  SKIP_CLIPS     : ${SKIP_CLIPS}"
echo "  FORCE_REBUILD  : ${FORCE_REBUILD}"
echo "============================================================"

# ---- Step 0: 切原子 clips ----
if [[ "${SKIP_CLIPS}" != "true" ]]; then
  echo ""
  echo "=== Step 0: Preparing atomic clips ==="

  CMD=(
    python3 "${REPO_ROOT}/proxy_data/youcook2_seg/prepare_all_clips.py"
    --annotation-dir "${ANNOTATION_DIR}"
    --output-dir "${CLIP_ROOT}"
    --levels L1 L2 L3
    --l1-fps "${L1_FPS}"
    --l2l3-fps "${L2L3_FPS}"
    --workers "${WORKERS}"
    --min-phases "${MIN_PHASES}" --max-phases "${MAX_PHASES}"
    --min-events "${MIN_EVENTS}" --max-events "${MAX_EVENTS}"
    --min-actions "${MIN_ACTIONS}" --max-actions "${MAX_ACTIONS}"
    --complete-only
  )
  if [[ -n "${SOURCE_VIDEO_DIR}" ]]; then
    CMD+=(--source-video-dir "${SOURCE_VIDEO_DIR}")
  fi
  "${CMD[@]}"
else
  echo ""
  echo "=== Step 0: Skipped (SKIP_CLIPS=true) ==="
fi

# ---- Step 1a: 构造 V2T 数据 ----
_build_data() {
  local exp_name="$1"
  shift
  local tasks=("$@")
  local out_dir="${OUTPUT_ROOT}/${exp_name}"

  if [[ "${FORCE_REBUILD}" != "true" && -f "${out_dir}/train.jsonl" ]]; then
    echo "  ${exp_name} already exists ($(wc -l < "${out_dir}/train.jsonl") train), skipping. Set FORCE_REBUILD=true to regenerate."
    return
  fi

  # 如果强制重建，先备份旧文件
  if [[ "${FORCE_REBUILD}" == "true" && -f "${out_dir}/train.jsonl" ]]; then
    local _bak="${out_dir}/train.jsonl.bak.$(date +%Y%m%d_%H%M%S)"
    echo "  ${exp_name}: backing up old train.jsonl -> ${_bak}"
    mv "${out_dir}/train.jsonl" "${_bak}"
    [[ -f "${out_dir}/val.jsonl" ]] && mv "${out_dir}/val.jsonl" "${out_dir}/val.jsonl.bak.$(date +%Y%m%d_%H%M%S)"
  fi

  python3 "${REPO_ROOT}/proxy_data/youcook2_seg/temporal_aot/build_aot_from_seg.py" \
    --annotation-dir "${ANNOTATION_DIR}" \
    --clip-dir "${CLIP_ROOT}" \
    --output-dir "${out_dir}" \
    --concat-dir "${out_dir}/concat_videos" \
    --concat-workers "${CONCAT_WORKERS}" \
    --tasks "${tasks[@]}" \
    --min-phases "${MIN_PHASES}" --max-phases "${MAX_PHASES}" \
    --min-events "${MIN_EVENTS}" --max-events "${MAX_EVENTS}" \
    --min-actions "${MIN_ACTIONS}" --max-actions "${MAX_ACTIONS}" \
    --train-total "${TRAIN_TOTAL}" \
    --level-ratio "${LEVEL_RATIO}" \
    --total-val "${TOTAL_VAL}" \
    --seed "${SEED}" --complete-only
}

echo ""
echo "=== Step 1a: Building V2T data ==="
_build_data "seg_aot_v2t" phase_v2t event_v2t action_v2t

echo ""
echo "=== Step 1b: Building T2V data ==="
_build_data "seg_aot_t2v" phase_t2v event_t2v action_t2v

# ---- 数据统计 ----
echo ""
echo "=== Data Summary ==="
for exp in seg_aot_v2t seg_aot_t2v; do
  _dir="${OUTPUT_ROOT}/${exp}"
  if [[ -f "${_dir}/train.jsonl" ]]; then
    echo "  ${exp}: $(wc -l < "${_dir}/train.jsonl") train + $(wc -l < "${_dir}/val.jsonl") val"
    if [[ -f "${_dir}/stats.json" ]]; then
      python3 -c "
import json
s = json.load(open('${_dir}/stats.json'))
for k, v in sorted(s.get('train_by_type', {}).items()):
    print(f'    {k}: {v}')
"
    fi
  else
    echo "  ${exp}: NOT BUILT"
  fi
done

echo ""
echo "=== Done ==="
echo "V2T: ${OUTPUT_ROOT}/seg_aot_v2t/train.jsonl"
echo "T2V: ${OUTPUT_ROOT}/seg_aot_t2v/train.jsonl"
