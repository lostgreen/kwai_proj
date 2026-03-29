#!/usr/bin/env bash
# =============================================================
# exp_pa1_original.sh — Prompt Ablation 实验 1: 原始标注 Prompt
#
# 使用 build_hier_data.py 生成的原始提示词（来自 prompts.py）:
#   L2: "Detect all complete cooking events..."
#   L3: "Detect all atomic cooking actions..."
#
# 特点: 含领域词汇 (cooking/recipe)，无稀疏采样约束，语义描述式
#
# 用法:
#   bash exp_pa1_original.sh
#   MAX_STEPS=30 bash exp_pa1_original.sh
# =============================================================
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_EXP_DIR="${SCRIPT_DIR}"
source "${_EXP_DIR}/../common.sh"

# ---- 实验命名 ----
EXP_NAME="${EXP_NAME:-prompt_ablation_PA1_original}"

# ---- 数据: L2+L3 (跳过 L1 warped 问题) ----
LEVELS="L2 L3"
TRAIN_PER_LEVEL="${TRAIN_PER_LEVEL:-400}"
VAL_PER_LEVEL="${VAL_PER_LEVEL:-100}"

DATA_DIR="${ABLATION_DATA_ROOT}/${EXP_NAME}"
TRAIN_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/val.jsonl"

# ---- 基础数据路径 (build_hier_data.py 输出，含原始 prompt) ----
_LEVELS_TAG="$(echo "${LEVELS}" | tr ' ' '_')"
BASE_DATA_DIR="${ABLATION_DATA_ROOT}/hier_seg_base_${_LEVELS_TAG}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  # Step 1: 构建基础数据（原始 prompt 由 build_hier_data.py 内调用 prompts.py 生成）
  if [[ ! -f "${BASE_DATA_DIR}/train.jsonl" ]]; then
    echo "[prompt_ablation] Step 1: Building base data from annotations (original prompts) ..."
    BUILD_LEVELS="${LEVELS//L3/L3_seg}"
    # shellcheck disable=SC2086
    python3 "${_EXP_DIR}/../build_hier_data.py" \
      --annotation-dir "${ANNOTATION_DIR}" \
      --clip-dir-l2 "${CLIP_DIR_L2}" \
      --clip-dir-l3 "${CLIP_DIR_L3}" \
      --output-dir "${BASE_DATA_DIR}" \
      --levels ${BUILD_LEVELS} \
      --total-val "$(( VAL_PER_LEVEL * 2 ))" \
      --train-per-level "${TRAIN_PER_LEVEL}" \
      --complete-only
  fi

  # Step 2: 直接采样，不替换 prompt (保留原始标注 prompt)
  echo "[prompt_ablation] Step 2: Sampling from base data (keeping original prompts) ..."
  mkdir -p "${DATA_DIR}"

  python3 -c "
import json, random, sys
from collections import Counter

random.seed(42)

records = []
for path in ['${BASE_DATA_DIR}/train.jsonl', '${BASE_DATA_DIR}/val.jsonl']:
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except FileNotFoundError:
        pass

if not records:
    print('ERROR: No records found', file=sys.stderr)
    sys.exit(1)

# 按 problem_type 分层
by_level = {}
for r in records:
    pt = r.get('problem_type', '')
    by_level.setdefault(pt, []).append(r)

all_train, all_val = [], []
for pt in sorted(by_level):
    recs = by_level[pt]
    random.shuffle(recs)
    n_val = min(${VAL_PER_LEVEL}, len(recs) // 5)
    n_train = min(${TRAIN_PER_LEVEL}, len(recs) - n_val) if ${TRAIN_PER_LEVEL} > 0 else len(recs) - n_val
    all_val.extend(recs[:n_val])
    all_train.extend(recs[n_val:n_val + n_train])
    print(f'  {pt}: {n_train} train + {n_val} val (total={len(recs)})')

random.shuffle(all_train)
random.shuffle(all_val)

def _norm(r):
    m = r.get('metadata')
    if isinstance(m, dict):
        r = dict(r)
        m = dict(m)
        if 'level' in m: m['level'] = str(m['level'])
        for k in ('duration','n_events','n_segments','window_start','window_end','n_warped_frames','original_duration'):
            if k in m and m[k] is not None: m[k] = float(m[k])
        r['metadata'] = m
    return r

with open('${TRAIN_FILE}', 'w') as f:
    for r in all_train:
        f.write(json.dumps(_norm(r), ensure_ascii=False) + '\n')
with open('${TEST_FILE}', 'w') as f:
    for r in all_val:
        f.write(json.dumps(_norm(r), ensure_ascii=False) + '\n')

print(f'Total: {len(all_train)} train + {len(all_val)} val')
"
fi

# ---- 启动训练 ----
source "${_EXP_DIR}/../launch_train.sh"
