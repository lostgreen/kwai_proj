#!/usr/bin/env bash
# Chain-Seg 消融: 批量运行 V1 → V2
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

VARIANTS="${VARIANTS:-V1 V2}"
MAX_STEPS="${MAX_STEPS:-60}"

_START=$(date +%s)
echo "[chain_seg] Starting batch: variants=${VARIANTS}, max_steps=${MAX_STEPS}"

for VARIANT in ${VARIANTS}; do
  echo "[chain_seg] Running VARIANT=${VARIANT}"
  MAX_STEPS="${MAX_STEPS}" \
  VARIANT="${VARIANT}" \
  bash "${SCRIPT_DIR}/exp_chain_ablation.sh" "${VARIANT}"
  echo "[chain_seg] Completed VARIANT=${VARIANT} at $(date)"
done

_END=$(date +%s)
echo "[chain_seg] All variants completed. Elapsed: $(( (_END - _START) / 60 ))m"
