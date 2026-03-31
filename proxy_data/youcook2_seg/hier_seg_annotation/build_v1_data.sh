#!/usr/bin/env bash
# =============================================================
# build_v1_data.sh — V1 层级分割训练数据一键构建
#
# 位置: proxy_data/youcook2_seg/hier_seg_annotation/
# 独立于消融实验目录, 构建通用基础训练数据.
#
# 流程:
#   1. build_hier_data.py (per-phase L2, 筛选, 均衡采样, 生成 JSONL)
#   2. prepare_clips.py (L1@1fps, L2@2fps, L3@2fps 切 clip)
#   3. 合并所有层级到 train_all.jsonl / val_all.jsonl
#
# 用法:
#   bash build_v1_data.sh [--use-hint]
#
#   # 自定义参数:
#   BALANCE_PER_LEVEL=600 L2_MIN_EVENTS=3 bash build_v1_data.sh
# =============================================================

set -euo pipefail

# ---- 路径定位 ----
HIER_SEG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${HIER_SEG_DIR}/../../.." && pwd)"
ABLATION_DIR="${REPO_ROOT}/local_scripts/hier_seg_ablations"

# ---- 标注 & 输出目录 ----
ANNOTATION_DIR="${ANNOTATION_DIR:-/m2v_intern/xuboshen/zgw/data/hier_seg_annotation/annotations}"
DATA_ROOT="${DATA_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_seg}"
VERSION="${VERSION:-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/${VERSION}}"
CLIP_DIR="${OUTPUT_DIR}/clips"

# ---- 筛选参数 (与 visualize_annotations.py FilterConfig 对齐) ----
L1_MIN_PHASES="${L1_MIN_PHASES:-2}"
L1_MAX_PHASES="${L1_MAX_PHASES:-6}"
L2_MIN_EVENTS="${L2_MIN_EVENTS:-2}"
L2_MAX_EVENTS="${L2_MAX_EVENTS:-999}"
L3_MIN_ACTIONS="${L3_MIN_ACTIONS:-3}"
L3_MAX_ACTIONS="${L3_MAX_ACTIONS:-999}"

# ---- 采样 ----
TRAIN_PER_LEVEL="${TRAIN_PER_LEVEL:-800}"
TOTAL_VAL="${TOTAL_VAL:-300}"
BALANCE_PER_LEVEL="${BALANCE_PER_LEVEL:-800}"

# ---- L2 模式 ----
L2_MODE="${L2_MODE:-phase}"

# ---- hint 参数 ----
USE_HINT_FLAG=""
if [[ "${1:-}" == "--use-hint" ]] || [[ "${USE_HINT:-}" == "true" ]]; then
    USE_HINT_FLAG="--use-hint"
    echo "[build_v1] Hint mode: ON"
fi

# ---- clip 子目录 ----
CLIP_DIR_L1="${CLIP_DIR}/L1"
CLIP_DIR_L2="${CLIP_DIR}/L2"
CLIP_DIR_L3="${CLIP_DIR}/L3"

echo "========================================"
echo "  Build V1 Hier Seg Training Data"
echo "========================================"
echo "  VERSION:         ${VERSION}"
echo "  ANNOTATION_DIR:  ${ANNOTATION_DIR}"
echo "  OUTPUT_DIR:      ${OUTPUT_DIR}"
echo "  L2_MODE:         ${L2_MODE}"
echo "  FILTER:          L1=[${L1_MIN_PHASES},${L1_MAX_PHASES}] L2=[${L2_MIN_EVENTS},${L2_MAX_EVENTS}] L3=[${L3_MIN_ACTIONS},${L3_MAX_ACTIONS}]"
echo "  BALANCE:         ${BALANCE_PER_LEVEL}/level"
echo "  TRAIN_PER_LEVEL: ${TRAIN_PER_LEVEL}"
echo "  USE_HINT:        ${USE_HINT_FLAG:-OFF}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

# ---- Step 1: 构建 JSONL ----
echo ""
echo "[Step 1/3] Building JSONL (build_hier_data.py) ..."

for LEVEL in L1 L2 L3_seg; do
    echo "  → Building ${LEVEL} ..."
    python "${ABLATION_DIR}/build_hier_data.py" \
        --annotation-dir "${ANNOTATION_DIR}" \
        --output-dir "${OUTPUT_DIR}/${LEVEL}" \
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
done

# ---- Step 2: 切 clip ----
echo ""
echo "[Step 2/3] Preparing clips (prepare_clips.py) ..."

for LEVEL in L1 L2 L3_seg; do
    TRAIN_JSONL="${OUTPUT_DIR}/${LEVEL}/train.jsonl"
    VAL_JSONL="${OUTPUT_DIR}/${LEVEL}/val.jsonl"

    if [[ "${LEVEL}" == "L1" ]]; then
        LEVEL_CLIP_DIR="${CLIP_DIR_L1}"
        FPS_ARG="--l1-fps 1"
    elif [[ "${LEVEL}" == "L2" ]]; then
        LEVEL_CLIP_DIR="${CLIP_DIR_L2}"
        FPS_ARG="--l2l3-fps 2"
    else
        LEVEL_CLIP_DIR="${CLIP_DIR_L3}"
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
            --clip-dir "${LEVEL_CLIP_DIR}" \
            --workers 8 \
            ${FPS_ARG} \
            --overwrite
    done
done

# ---- Step 3: 合并所有层级 ----
echo ""
echo "[Step 3/3] Merging all levels into final train/val JSONL ..."
FINAL_TRAIN="${OUTPUT_DIR}/train_all.jsonl"
FINAL_VAL="${OUTPUT_DIR}/val_all.jsonl"

> "${FINAL_TRAIN}"
> "${FINAL_VAL}"

for LEVEL in L1 L2 L3_seg; do
    CLIPPED_TRAIN="${OUTPUT_DIR}/${LEVEL}/train_clipped.jsonl"
    CLIPPED_VAL="${OUTPUT_DIR}/${LEVEL}/val_clipped.jsonl"
    [[ -f "${CLIPPED_TRAIN}" ]] && cat "${CLIPPED_TRAIN}" >> "${FINAL_TRAIN}"
    [[ -f "${CLIPPED_VAL}" ]]   && cat "${CLIPPED_VAL}"   >> "${FINAL_VAL}"
done

TRAIN_COUNT=$(wc -l < "${FINAL_TRAIN}")
VAL_COUNT=$(wc -l < "${FINAL_VAL}")

echo ""
echo "========================================"
echo "  Build Complete!"
echo "  Train: ${TRAIN_COUNT} records → ${FINAL_TRAIN}"
echo "  Val:   ${VAL_COUNT} records → ${FINAL_VAL}"
echo "========================================"
