#!/usr/bin/env bash
# =============================================================
# run_sort_ablation.sh — 串行跑 L2 + L3 sort 消融
#
# 用法:
#   bash local_scripts/event_logic_ablations/run_sort_ablation.sh
# =============================================================
set -euo pipefail

_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo " Sort Ablation: L2 event sort → L3 action sort"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ---- Exp 1: L2 event sort ----
echo ""
echo ">>> [1/2] L2 Event Sort (exp1)"
echo ""
bash "${_dir}/exp1_sort.sh"

# ---- Exp 2: L3 action sort ----
echo ""
echo ">>> [2/2] L3 Action Sort (exp2)"
echo ""
bash "${_dir}/exp2_sort_l3.sh"

echo ""
echo "============================================================"
echo " Sort Ablation complete. $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
