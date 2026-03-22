#!/usr/bin/env bash
# =============================================================
# merge_checkpoints.sh — 批量 merge exp1-exp4 最后一个 checkpoint
#
# 输出格式：{OUTPUT_DIR}/{BASE_MODEL_NAME}_{EXP_NAME}
#
# 用法：
#   bash merge_checkpoints.sh
#
#   覆盖默认路径示例：
#   CHECKPOINT_ROOT=/other/path OUTPUT_DIR=/my/save/dir bash merge_checkpoints.sh
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# ---- 路径配置（与 common.sh 保持一致，可通过环境变量覆盖）----
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/youcook2_aot/ablations}"
MODEL_PATH="${MODEL_PATH:-/home/xuboshen/models/Qwen3-VL-4B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/ablation_merged}"

# Base model 名称（取路径最后一段）
BASE_MODEL_NAME="$(basename "${MODEL_PATH}")"

# ---- 需要 merge 的实验列表 ----
EXPS=(
    "aot_ablation_exp1_v2t_binary"
    "aot_ablation_exp2_v2t_3way"
    "aot_ablation_exp3_t2v_binary"
    "aot_ablation_exp4_t2v_3way"
)

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "CHECKPOINT_ROOT : ${CHECKPOINT_ROOT}"
echo "BASE_MODEL      : ${MODEL_PATH}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "=================================================="

for EXP_NAME in "${EXPS[@]}"; do
    EXP_CKPT_DIR="${CHECKPOINT_ROOT}/${EXP_NAME}"

    if [[ ! -d "${EXP_CKPT_DIR}" ]]; then
        echo "[SKIP] ${EXP_NAME}: checkpoint dir not found (${EXP_CKPT_DIR})"
        continue
    fi

    # 找最后一个 global_step_XXX 目录（按数字排序取最大）
    LAST_STEP_DIR="$(
        find "${EXP_CKPT_DIR}" -maxdepth 1 -type d -name 'global_step_*' \
        | sort -V \
        | tail -n 1
    )"

    if [[ -z "${LAST_STEP_DIR}" ]]; then
        echo "[SKIP] ${EXP_NAME}: no global_step_* directory found"
        continue
    fi

    ACTOR_DIR="${LAST_STEP_DIR}/actor"
    if [[ ! -d "${ACTOR_DIR}" ]]; then
        echo "[SKIP] ${EXP_NAME}: actor dir not found in ${LAST_STEP_DIR}"
        continue
    fi

    STEP_TAG="$(basename "${LAST_STEP_DIR}")"
    DEST_NAME="${BASE_MODEL_NAME}_${EXP_NAME}"
    DEST_DIR="${OUTPUT_DIR}/${DEST_NAME}"

    echo ""
    echo "[INFO] ===== ${EXP_NAME} ====="
    echo "[INFO] Last checkpoint : ${STEP_TAG}"
    echo "[INFO] Actor dir       : ${ACTOR_DIR}"
    echo "[INFO] Output dir      : ${DEST_DIR}"

    # 若目标目录已存在则跳过（删掉该判断则重新 merge）
    if [[ -d "${DEST_DIR}" ]]; then
        echo "[SKIP] ${DEST_NAME} already exists, skipping. Remove to re-merge."
        continue
    fi

    # ---- 执行 merge ----
    python3 "${REPO_ROOT}/scripts/model_merger.py" \
        --local_dir "${ACTOR_DIR}" \
        --base_model "${MODEL_PATH}"

    # merge 产出在 actor/huggingface/，移动到统一输出目录
    HF_DIR="${ACTOR_DIR}/huggingface"
    if [[ ! -d "${HF_DIR}" ]]; then
        echo "[ERROR] merge did not produce ${HF_DIR}, skipping copy"
        continue
    fi

    cp -r "${HF_DIR}" "${DEST_DIR}"
    echo "[DONE] Saved to: ${DEST_DIR}"
done

echo ""
echo "=================================================="
echo "All done. Merged models in: ${OUTPUT_DIR}"
ls -1 "${OUTPUT_DIR}"
echo "=================================================="
