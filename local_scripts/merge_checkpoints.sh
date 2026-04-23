#!/usr/bin/env bash
# =============================================================
# merge_checkpoints.sh — 通用批量 merge ablation checkpoint
#
# 自动遍历 CHECKPOINT_ROOT 下所有含 global_step_* 的实验子目录，
# merge 最后一个 checkpoint，并生成 model_meta.py 配置文件。
#
# 用法：
#   bash merge_checkpoints.sh <CHECKPOINT_ROOT> [OUTPUT_DIR]
#
#   也可通过环境变量覆盖：
#   MODEL_PATH=/path/to/base bash merge_checkpoints.sh /path/to/ablations
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# ---- 参数解析 ----
if [[ $# -lt 1 ]]; then
    echo "Usage: bash merge_checkpoints.sh <CHECKPOINT_ROOT> [OUTPUT_DIR]"
    echo ""
    echo "  CHECKPOINT_ROOT  ablations 目录，下面每个子目录是一个实验"
    echo "  OUTPUT_DIR       merge 输出目录（默认: CHECKPOINT_ROOT_merged）"
    exit 1
fi

CHECKPOINT_ROOT="$(cd -- "$1" && pwd)"
OUTPUT_DIR="${2:-${CHECKPOINT_ROOT%/}_merged}"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"

# Base model 名称（取路径最后一段）
BASE_MODEL_NAME="$(basename "${MODEL_PATH}")"

# meta 配置文件
META_FILE="${OUTPUT_DIR}/model_meta.py"

mkdir -p "${OUTPUT_DIR}"

echo "=================================================="
echo "CHECKPOINT_ROOT : ${CHECKPOINT_ROOT}"
echo "BASE_MODEL      : ${MODEL_PATH}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "META_FILE       : ${META_FILE}"
echo "=================================================="

# ---- 收集成功 merge 的条目，用于生成 meta ----
declare -a META_ENTRIES=()

# ---- 自动发现实验目录 ----
for EXP_CKPT_DIR in "${CHECKPOINT_ROOT}"/*/; do
    # 去掉尾部的 /
    EXP_CKPT_DIR="${EXP_CKPT_DIR%/}"
    EXP_NAME="$(basename "${EXP_CKPT_DIR}")"

    # 找最后一个 global_step_XXX 目录（按数字排序取最大）
    LAST_STEP_DIR="$(
        find "${EXP_CKPT_DIR}" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
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
        META_ENTRIES+=("${DEST_NAME}|${DEST_DIR}")
        continue
    fi

    # ---- 执行 merge ----
    python3 "${REPO_ROOT}/scripts/model_merger.py" \
        --local_dir "${ACTOR_DIR}" \
        --base_model "${MODEL_PATH}"

    # merge 产出在 actor/huggingface/，复制到统一输出目录
    HF_DIR="${ACTOR_DIR}/huggingface"
    if [[ ! -d "${HF_DIR}" ]]; then
        echo "[ERROR] merge did not produce ${HF_DIR}, skipping copy"
        continue
    fi

    cp -r "${HF_DIR}" "${DEST_DIR}"
    echo "[DONE] Saved to: ${DEST_DIR}"
    META_ENTRIES+=("${DEST_NAME}|${DEST_DIR}")
done

# ---- 生成 model_meta.py ----
{
    echo "from functools import partial"
    echo "from vlmeval.vlm import Qwen3VLChat"
    echo ""
    echo ""
    echo "ABLATION_MODELS = {"
    for entry in "${META_ENTRIES[@]}"; do
        NAME="${entry%%|*}"
        DIR="${entry##*|}"
        cat <<PYEOF
    "${NAME}": partial(
        Qwen3VLChat,
        model_path="${DIR}",
        use_custom_prompt=False,
        use_vllm=True,
        temperature=0,
        max_new_tokens=16384,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        top_p=0.8,
        top_k=20,
    ),
PYEOF
    done
    echo "}"
} > "${META_FILE}"

echo ""
echo "=================================================="
echo "All done. Merged models in: ${OUTPUT_DIR}"
ls -1 "${OUTPUT_DIR}"
echo ""
echo "Meta file written to: ${META_FILE}"
echo "=================================================="
