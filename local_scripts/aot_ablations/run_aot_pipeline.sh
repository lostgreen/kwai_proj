#!/usr/bin/env bash
# =============================================================
# run_aot_pipeline.sh — 一键构造 Seg-AOT V2T/T2V 消融数据 + demo run
#
# 用法:
#   # 全流程: 切 clips → 构造数据 → 2 卡 demo run
#   bash local_scripts/aot_ablations/run_aot_pipeline.sh
#
#   # 跳过 clips (已切好)
#   SKIP_CLIPS=true bash local_scripts/aot_ablations/run_aot_pipeline.sh
#
#   # 只构造数据不跑训练
#   SKIP_TRAIN=true bash local_scripts/aot_ablations/run_aot_pipeline.sh
#
#   # 自定义参数
#   TRAIN_TOTAL=500 LEVEL_RATIO=1:1:1 MAX_PHASES=5 \
#     bash local_scripts/aot_ablations/run_aot_pipeline.sh
# =============================================================
set -euo pipefail
set -x

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SKIP_CLIPS="${SKIP_CLIPS:-false}"
SKIP_TRAIN="${SKIP_TRAIN:-true}"    # 默认只构造数据不跑训练
DEMO_GPUS="${DEMO_GPUS:-2}"

DATA_ROOT="${SEG_AOT_DATA_ROOT}"

echo "============================================================"
echo "Seg-AOT Pipeline"
echo "  ANNOTATION_DIR : ${ANNOTATION_DIR}"
echo "  CLIP_ROOT      : ${CLIP_ROOT}"
echo "  DATA_ROOT      : ${DATA_ROOT}"
echo "  TRAIN_TOTAL    : ${TRAIN_TOTAL}"
echo "  LEVEL_RATIO    : ${LEVEL_RATIO} (L1:L2:L3)"
echo "  FILTER         : phases=[${MIN_PHASES},${MAX_PHASES}] events=[${MIN_EVENTS},${MAX_EVENTS}] actions=[${MIN_ACTIONS},${MAX_ACTIONS}]"
echo "  SKIP_CLIPS     : ${SKIP_CLIPS}"
echo "  SKIP_TRAIN     : ${SKIP_TRAIN}"
echo "============================================================"

# ---- Step 0: 切原子 clips ----
if [[ "${SKIP_CLIPS}" != "true" ]]; then
  echo ""
  echo "=== Step 0: Preparing atomic clips ==="
  bash "${SCRIPT_DIR}/prepare_clips.sh"
else
  echo ""
  echo "=== Step 0: Skipped (SKIP_CLIPS=true) ==="
fi

# ---- Step 1: 构造 V2T 数据 ----
echo ""
echo "=== Step 1a: Building V2T data ==="
V2T_DIR="${DATA_ROOT}/seg_aot_v2t"
if [[ -f "${V2T_DIR}/train.jsonl" ]]; then
  echo "  V2T data already exists at ${V2T_DIR}/train.jsonl, skipping."
else
  python3 "${REPO_ROOT}/proxy_data/youcook2_seg/temporal_aot/build_aot_from_seg.py" \
    --annotation-dir "${ANNOTATION_DIR}" \
    --clip-dir "${CLIP_ROOT}" \
    --output-dir "${V2T_DIR}" \
    --concat-dir "${V2T_DIR}/concat_videos" \
    --concat-workers "${CONCAT_WORKERS}" \
    --tasks phase_v2t event_v2t action_v2t \
    --min-phases "${MIN_PHASES}" --max-phases "${MAX_PHASES}" \
    --min-events "${MIN_EVENTS}" --max-events "${MAX_EVENTS}" \
    --min-actions "${MIN_ACTIONS}" --max-actions "${MAX_ACTIONS}" \
    --train-total "${TRAIN_TOTAL}" \
    --level-ratio "${LEVEL_RATIO}" \
    --total-val "${TOTAL_VAL}" \
    --seed 42 --complete-only
fi

# ---- Step 1b: 构造 T2V 数据 ----
echo ""
echo "=== Step 1b: Building T2V data ==="
T2V_DIR="${DATA_ROOT}/seg_aot_t2v"
if [[ -f "${T2V_DIR}/train.jsonl" ]]; then
  echo "  T2V data already exists at ${T2V_DIR}/train.jsonl, skipping."
else
  python3 "${REPO_ROOT}/proxy_data/youcook2_seg/temporal_aot/build_aot_from_seg.py" \
    --annotation-dir "${ANNOTATION_DIR}" \
    --clip-dir "${CLIP_ROOT}" \
    --output-dir "${T2V_DIR}" \
    --concat-dir "${T2V_DIR}/concat_videos" \
    --concat-workers "${CONCAT_WORKERS}" \
    --tasks phase_t2v event_t2v action_t2v \
    --min-phases "${MIN_PHASES}" --max-phases "${MAX_PHASES}" \
    --min-events "${MIN_EVENTS}" --max-events "${MAX_EVENTS}" \
    --min-actions "${MIN_ACTIONS}" --max-actions "${MAX_ACTIONS}" \
    --train-total "${TRAIN_TOTAL}" \
    --level-ratio "${LEVEL_RATIO}" \
    --total-val "${TOTAL_VAL}" \
    --seed 42 --complete-only
fi

# ---- 数据统计 ----
echo ""
echo "=== Data Summary ==="
for exp in seg_aot_v2t seg_aot_t2v; do
  _dir="${DATA_ROOT}/${exp}"
  if [[ -f "${_dir}/train.jsonl" ]]; then
    _n_train=$(wc -l < "${_dir}/train.jsonl")
    _n_val=$(wc -l < "${_dir}/val.jsonl")
    echo "  ${exp}: ${_n_train} train + ${_n_val} val"
    if [[ -f "${_dir}/stats.json" ]]; then
      python3 -c "
import json
s = json.load(open('${_dir}/stats.json'))
for k, v in s.get('train_by_type', {}).items():
    print(f'    {k}: {v}')
"
    fi
  else
    echo "  ${exp}: NOT BUILT"
  fi
done

# ---- Step 2: Demo train (可选) ----
if [[ "${SKIP_TRAIN}" == "true" ]]; then
  echo ""
  echo "=== Step 2: Skipped (SKIP_TRAIN=true) ==="
  echo ""
  echo "To run 2-GPU demo:"
  echo "  SKIP_TRAIN=false DEMO_GPUS=2 bash local_scripts/aot_ablations/run_aot_pipeline.sh"
  echo ""
  echo "Or manually:"
  echo "  N_GPUS_PER_NODE=2 TP_SIZE=1 ROLLOUT_BS=4 GLOBAL_BS=4 ROLLOUT_N=4 \\"
  echo "  MAX_STEPS=10 SKIP_CLIPS=true DATA_DIR=${V2T_DIR} EXP_NAME=seg_aot_v2t_demo \\"
  echo "  SEG_TASKS='phase_v2t event_v2t action_v2t' bash local_scripts/aot_ablations/launch_seg_train.sh"
  exit 0
fi

echo ""
echo "=== Step 2: Demo training (${DEMO_GPUS} GPUs) ==="

N_GPUS_PER_NODE="${DEMO_GPUS}" \
TP_SIZE=1 \
ROLLOUT_BS=4 \
GLOBAL_BS=4 \
ROLLOUT_N=4 \
MAX_STEPS=10 \
SAVE_FREQ=5 \
VAL_FREQ=5 \
SKIP_CLIPS=true \
DATA_DIR="${V2T_DIR}" \
EXP_NAME="seg_aot_v2t_demo" \
SEG_TASKS="phase_v2t event_v2t action_v2t" \
bash "${SCRIPT_DIR}/launch_seg_train.sh"
