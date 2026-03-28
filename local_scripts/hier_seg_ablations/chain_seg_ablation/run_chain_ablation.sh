#!/usr/bin/env bash
# Chain-Seg: 运行 V2 (ground-seg) 实验
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MAX_STEPS="${MAX_STEPS:-60}"

echo "[chain_seg] Starting V2 (ground-seg), max_steps=${MAX_STEPS}"

MAX_STEPS="${MAX_STEPS}" \
bash "${SCRIPT_DIR}/exp_chain_ablation.sh"

echo "[chain_seg] Completed at $(date)"
