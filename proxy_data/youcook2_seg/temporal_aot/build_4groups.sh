#!/usr/bin/env bash
# =============================================================
# build_4groups.sh — 构造 4 组 AOT 实验数据 (v2t/t2v × binary/3way)
#
# 每组独立的 train.jsonl + val.jsonl, 各含 event + action 两种粒度.
#
# 输出:
#   v2t_binary/  — event_v2t_binary + action_v2t_binary
#   t2v_binary/  — event_t2v_binary + action_t2v_binary
#   v2t_3way/    — event_v2t_3way   + action_v2t_3way
#   t2v_3way/    — event_t2v_3way   + action_t2v_3way
#
# 用法:
#   bash proxy_data/youcook2_seg/temporal_aot/build_4groups.sh
#
#   # 自定义参数
#   TRAIN_TOTAL=500 ANNOTATION_DIR=/path/to/fixed \
#     bash proxy_data/youcook2_seg/temporal_aot/build_4groups.sh
#
#   # 跳过 clip 切分 (已切好)
#   SKIP_CLIPS=true bash proxy_data/youcook2_seg/temporal_aot/build_4groups.sh
# =============================================================
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# ---- 路径 ----
HIER_SEG_ROOT="${HIER_SEG_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation}"
ANNOTATION_DIR="${ANNOTATION_DIR:-${HIER_SEG_ROOT}/annotations_fixed_gmn25}"
CLIP_ROOT="${CLIP_ROOT:-${HIER_SEG_ROOT}/clips}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_seg_aot}"
SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-}"

# ---- 过滤参数 ----
MIN_EVENTS="${MIN_EVENTS:-3}"
MAX_EVENTS="${MAX_EVENTS:-8}"
MIN_ACTIONS="${MIN_ACTIONS:-3}"
MAX_ACTIONS="${MAX_ACTIONS:-10}"

# ---- 采样参数 ----
TRAIN_TOTAL="${TRAIN_TOTAL:-1000}"
LEVEL_RATIO="${LEVEL_RATIO:-1:1}"
TOTAL_VAL="${TOTAL_VAL:-200}"
SEED="${SEED:-42}"

# ---- Clip 参数 ----
L2L3_FPS="${L2L3_FPS:-2}"
WORKERS="${WORKERS:-8}"
CONCAT_WORKERS="${CONCAT_WORKERS:-8}"

SKIP_CLIPS="${SKIP_CLIPS:-false}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
FILTER_ORDER="${FILTER_ORDER:-false}"

echo "============================================================"
echo "AOT 4-Group Builder (v2t/t2v × binary/3way)"
echo "  ANNOTATION_DIR : ${ANNOTATION_DIR}"
echo "  CLIP_ROOT      : ${CLIP_ROOT}"
echo "  OUTPUT_ROOT    : ${OUTPUT_ROOT}"
echo "  TRAIN_TOTAL    : ${TRAIN_TOTAL} per group"
echo "  LEVEL_RATIO    : ${LEVEL_RATIO} (event:action)"
echo "  FILTER         : events=[${MIN_EVENTS},${MAX_EVENTS}] actions=[${MIN_ACTIONS},${MAX_ACTIONS}]"
echo "  SKIP_CLIPS     : ${SKIP_CLIPS}"
echo "  FORCE_REBUILD  : ${FORCE_REBUILD}"
echo "  FILTER_ORDER   : ${FILTER_ORDER}"
echo "============================================================"

# ---- Step 0: 切原子 clips (可选) ----
if [[ "${SKIP_CLIPS}" != "true" ]]; then
  echo ""
  echo "=== Step 0: Preparing atomic clips ==="

  CMD=(
    python3 "${REPO_ROOT}/proxy_data/youcook2_seg/prepare_all_clips.py"
    --annotation-dir "${ANNOTATION_DIR}"
    --output-dir "${CLIP_ROOT}"
    --levels L2 L3
    --l2l3-fps "${L2L3_FPS}"
    --workers "${WORKERS}"
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

# ---- Build helper ----
FILTER_ORDER_FLAG=""
if [[ "${FILTER_ORDER}" == "true" ]]; then
  FILTER_ORDER_FLAG="--filter-order"
fi

_build_group() {
  local group_name="$1"
  shift
  local tasks=("$@")
  local out_dir="${OUTPUT_ROOT}/${group_name}"

  echo ""
  echo "=== Building: ${group_name} (tasks: ${tasks[*]}) ==="

  if [[ "${FORCE_REBUILD}" != "true" && -f "${out_dir}/train.jsonl" ]]; then
    echo "  ${group_name} already exists ($(wc -l < "${out_dir}/train.jsonl") train), skipping."
    echo "  Set FORCE_REBUILD=true to regenerate."
    return
  fi

  # 备份旧文件
  if [[ "${FORCE_REBUILD}" == "true" && -f "${out_dir}/train.jsonl" ]]; then
    local _bak="${out_dir}/train.jsonl.bak.$(date +%Y%m%d_%H%M%S)"
    echo "  ${group_name}: backing up old train.jsonl -> ${_bak}"
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
    --min-events "${MIN_EVENTS}" --max-events "${MAX_EVENTS}" \
    --min-actions "${MIN_ACTIONS}" --max-actions "${MAX_ACTIONS}" \
    --train-total "${TRAIN_TOTAL}" \
    --level-ratio "${LEVEL_RATIO}" \
    --total-val "${TOTAL_VAL}" \
    --seed "${SEED}" --complete-only \
    ${FILTER_ORDER_FLAG}
}

# ---- 4 组构建 ----
_build_group "v2t_binary" event_v2t_binary action_v2t_binary
_build_group "t2v_binary" event_t2v_binary action_t2v_binary
_build_group "v2t_3way"   event_v2t_3way   action_v2t_3way
_build_group "t2v_3way"   event_t2v_3way   action_t2v_3way

# ---- 数据统计 ----
echo ""
echo "=== Data Summary ==="
for group in v2t_binary t2v_binary v2t_3way t2v_3way; do
  _dir="${OUTPUT_ROOT}/${group}"
  if [[ -f "${_dir}/train.jsonl" ]]; then
    echo "  ${group}: $(wc -l < "${_dir}/train.jsonl") train + $(wc -l < "${_dir}/val.jsonl") val"
    if [[ -f "${_dir}/stats.json" ]]; then
      python3 -c "
import json
s = json.load(open('${_dir}/stats.json'))
for k, v in sorted(s.get('train_by_type', {}).items()):
    print(f'    {k}: {v}')
"
    fi
  else
    echo "  ${group}: NOT BUILT"
  fi
done

echo ""
echo "=== Done ==="
for group in v2t_binary t2v_binary v2t_3way t2v_3way; do
  echo "  ${group}: ${OUTPUT_ROOT}/${group}/train.jsonl"
done
