#!/usr/bin/env bash
# =============================================================
# build_v1_data.sh — V1 层级分割训练数据一键构建
#
# 位置: proxy_data/youcook2_seg/hier_seg_annotation/
# 输出: 标注目录同级的 train/ 下, clips 在各层的 videos/ 子目录
#
# 输出结构 (无 hint):
#   ${OUTPUT_DIR}/
#   ├── L1/
#   │   ├── train.jsonl / val.jsonl / train_clipped.jsonl / val_clipped.jsonl
#   │   └── videos/      {clip_key}_L1_1fps.mp4
#   ├── L2/   (同上)
#   ├── L3_seg/  (同上)
#   ├── train_all.jsonl   (三层合并)
#   └── val_all.jsonl
#
# 输出结构 (--use-hint, 复用已有 videos/):
#   ${OUTPUT_DIR}/
#   ├── L1/
#   │   ├── train_hint.jsonl / val_hint.jsonl
#   │   ├── train_hint_clipped.jsonl / val_hint_clipped.jsonl
#   │   └── videos/      ← 复用, 不重新切
#   ├── L2/   (同上)
#   ├── L3_seg/  (同上)
#   ├── train_hint_all.jsonl
#   └── val_hint_all.jsonl
#
# 用法:
#   bash build_v1_data.sh               # 无 hint 版本 (首次运行, 切 clips)
#   bash build_v1_data.sh --use-hint    # hint 版本 (复用已有 clips)
#   BALANCE_PER_LEVEL=600 bash build_v1_data.sh
# =============================================================

set -euo pipefail

# ---- 路径定位 ----
HIER_SEG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HIER_SEG_DIR}/../../.." && pwd)"

# ---- 数据根目录 (annotations 所在目录) ----
DATA_ROOT="${DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation}"
ANNOTATION_DIR="${ANNOTATION_DIR:-${DATA_ROOT}/annotations}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/train}"

# ---- 筛选参数 (与 visualize_annotations.py 对齐) ----
L1_MIN_PHASES="${L1_MIN_PHASES:-2}"
L1_MAX_PHASES="${L1_MAX_PHASES:-6}"
L2_MIN_EVENTS="${L2_MIN_EVENTS:-3}"
L2_MAX_EVENTS="${L2_MAX_EVENTS:-8}"
L3_MIN_ACTIONS="${L3_MIN_ACTIONS:-3}"
L3_MAX_ACTIONS="${L3_MAX_ACTIONS:-10}"

# ---- 采样 ----
TRAIN_PER_LEVEL="${TRAIN_PER_LEVEL:-800}"
TOTAL_VAL="${TOTAL_VAL:-300}"
BALANCE_PER_LEVEL="${BALANCE_PER_LEVEL:-800}"

# ---- L2 模式 ----
L2_MODE="${L2_MODE:-phase}"

# ---- hint 参数 ----
USE_HINT=false
USE_HINT_FLAG=""
if [[ "${1:-}" == "--use-hint" ]] || [[ "${USE_HINT_ENV:-}" == "true" ]]; then
    USE_HINT=true
    USE_HINT_FLAG="--use-hint"
fi

# ---- 文件名后缀 (hint 模式用 _hint, 否则无后缀) ----
if ${USE_HINT}; then
    SUFFIX="_hint"
else
    SUFFIX=""
fi

echo "========================================"
echo "  Build V1 Hier Seg Training Data"
echo "========================================"
echo "  DATA_ROOT:       ${DATA_ROOT}"
echo "  ANNOTATION_DIR:  ${ANNOTATION_DIR}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  L2_MODE:         ${L2_MODE}"
echo "  FILTER:          L1=[${L1_MIN_PHASES},${L1_MAX_PHASES}] L2=[${L2_MIN_EVENTS},${L2_MAX_EVENTS}] L3=[${L3_MIN_ACTIONS},${L3_MAX_ACTIONS}]"
echo "  BALANCE:         ${BALANCE_PER_LEVEL}/level"
echo "  TRAIN_PER_LEVEL: ${TRAIN_PER_LEVEL}"
echo "  HINT:            ${USE_HINT} (suffix='${SUFFIX}')"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

# ---- Step 1: 构建 JSONL ----
echo ""
echo "[Step 1/3] Building JSONL (build_hier_data.py) ..."

for LEVEL in L1 L2 L3_seg; do
    # hint 模式: 输出到临时子目录, 然后重命名
    if ${USE_HINT}; then
        BUILD_OUT_DIR="${OUTPUT_DIR}/${LEVEL}/_hint_tmp"
    else
        BUILD_OUT_DIR="${OUTPUT_DIR}/${LEVEL}"
    fi

    echo "  → Building ${LEVEL}${SUFFIX} ..."
    python "${HIER_SEG_DIR}/build_hier_data.py" \
        --annotation-dir "${ANNOTATION_DIR}" \
        --output-dir "${BUILD_OUT_DIR}" \
        --levels "${LEVEL}" \
        --l2-mode "${L2_MODE}" \
        --l1-min-phases "${L1_MIN_PHASES}" \
        --l1-max-phases "${L1_MAX_PHASES}" \
        --l2-min-events "${L2_MIN_EVENTS}" \
        --l2-max-events "${L2_MAX_EVENTS}" \
        --l3-min-actions "${L3_MIN_ACTIONS}" \
        --l3-max-actions "${L3_MAX_ACTIONS}" \
        --complete-only \
        --balance-per-level "${BALANCE_PER_LEVEL}" \
        --train-per-level "${TRAIN_PER_LEVEL}" \
        --total-val "${TOTAL_VAL}" \
        ${USE_HINT_FLAG}

    # hint 模式: 将 train.jsonl/val.jsonl 重命名为 train_hint.jsonl/val_hint.jsonl
    if ${USE_HINT}; then
        for SPLIT in train val; do
            SRC="${BUILD_OUT_DIR}/${SPLIT}.jsonl"
            DST="${OUTPUT_DIR}/${LEVEL}/${SPLIT}${SUFFIX}.jsonl"
            if [[ -f "${SRC}" ]]; then
                mv "${SRC}" "${DST}"
                echo "    → ${DST}"
            fi
        done
        rm -rf "${BUILD_OUT_DIR}"
    fi
done

# ---- Step 2: 切 clip (videos/ 放在各层子目录下) ----
echo ""
echo "[Step 2/3] Preparing clips (prepare_clips.py) ..."

for LEVEL in L1 L2 L3_seg; do
    TRAIN_JSONL="${OUTPUT_DIR}/${LEVEL}/train${SUFFIX}.jsonl"
    VAL_JSONL="${OUTPUT_DIR}/${LEVEL}/val${SUFFIX}.jsonl"
    LEVEL_VIDEO_DIR="${OUTPUT_DIR}/${LEVEL}/videos"

    if [[ "${LEVEL}" == "L1" ]]; then
        FPS_ARG="--l1-fps 1"
    else
        FPS_ARG="--l2l3-fps 2"
    fi

    for SPLIT_JSONL in "${TRAIN_JSONL}" "${VAL_JSONL}"; do
        if [[ ! -f "${SPLIT_JSONL}" ]]; then
            echo "  SKIP: ${SPLIT_JSONL} not found"
            continue
        fi
        OUT_JSONL="${SPLIT_JSONL%.jsonl}_clipped.jsonl"
        echo "  → ${SPLIT_JSONL} → ${OUT_JSONL}"
        python "${HIER_SEG_DIR}/prepare_clips.py" \
            --input "${SPLIT_JSONL}" \
            --output "${OUT_JSONL}" \
            --clip-dir "${LEVEL_VIDEO_DIR}" \
            --workers 8 \
            ${FPS_ARG} \
            --overwrite
    done
done

# ---- Step 3: 合并所有层级 ----
echo ""
echo "[Step 3/3] Merging all levels into final train/val JSONL ..."
FINAL_TRAIN="${OUTPUT_DIR}/train${SUFFIX}_all.jsonl"
FINAL_VAL="${OUTPUT_DIR}/val${SUFFIX}_all.jsonl"

> "${FINAL_TRAIN}"
> "${FINAL_VAL}"

for LEVEL in L1 L2 L3_seg; do
    CLIPPED_TRAIN="${OUTPUT_DIR}/${LEVEL}/train${SUFFIX}_clipped.jsonl"
    CLIPPED_VAL="${OUTPUT_DIR}/${LEVEL}/val${SUFFIX}_clipped.jsonl"
    [[ -f "${CLIPPED_TRAIN}" ]] && cat "${CLIPPED_TRAIN}" >> "${FINAL_TRAIN}"
    [[ -f "${CLIPPED_VAL}" ]]   && cat "${CLIPPED_VAL}"   >> "${FINAL_VAL}"
done

TRAIN_COUNT=$(wc -l < "${FINAL_TRAIN}")
VAL_COUNT=$(wc -l < "${FINAL_VAL}")

echo ""
echo "========================================"
echo "  Build Complete!  (hint=${USE_HINT})"
echo "  Train: ${TRAIN_COUNT} records → ${FINAL_TRAIN}"
echo "  Val:   ${VAL_COUNT} records → ${FINAL_VAL}"
echo "========================================"
echo ""
echo "  输出结构:"
echo "  ${OUTPUT_DIR}/"
echo "  ├── L1/   (train${SUFFIX}.jsonl + val${SUFFIX}.jsonl + videos/)"
echo "  ├── L2/   (train${SUFFIX}.jsonl + val${SUFFIX}.jsonl + videos/)"
echo "  ├── L3_seg/ (train${SUFFIX}.jsonl + val${SUFFIX}.jsonl + videos/)"
echo "  ├── train${SUFFIX}_all.jsonl"
echo "  └── val${SUFFIX}_all.jsonl"
