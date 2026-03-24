#!/usr/bin/env bash
# =============================================================
# run_batch.sh — 一键运行 Hier Seg 消融实验（3 台机器分批跑）
#
# 用法:
#   bash local_scripts/hier_seg_ablations/run_batch.sh 1   # 机器 A: exp1 → exp2 → exp3
#   bash local_scripts/hier_seg_ablations/run_batch.sh 2   # 机器 B: exp4 → exp5
#   bash local_scripts/hier_seg_ablations/run_batch.sh 3   # 机器 C: exp6 → exp7
#
# 分配逻辑（按序号均分，所有实验互相独立）:
#   机器 A: exp1(L2_only) → exp2(L3_seq) → exp3(L3_shuf)
#   机器 B: exp4(L3_both) → exp5(L2_L3)
#   机器 C: exp6(L1_L2_L3) → exp7(all_mixed)
#
# 可覆盖:
#   MAX_STEPS=60  bash local_scripts/hier_seg_ablations/run_batch.sh 1
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---- 默认参数 ----
export MAX_STEPS="${MAX_STEPS:-60}"

MACHINE="${1:-}"
if [[ -z "${MACHINE}" || ! "${MACHINE}" =~ ^[1-3]$ ]]; then
  echo "Usage: bash $0 <machine_id:1|2|3>"
  echo ""
  echo "  1 → exp1, exp2, exp3   (机器 A)"
  echo "  2 → exp4, exp5          (机器 B)"
  echo "  3 → exp6, exp7          (机器 C)"
  exit 1
fi

# ---- 运行单个实验 ----
run_exp() {
  local exp_id="$1"
  local script="${SCRIPT_DIR}/exp${exp_id}_*.sh"
  # glob 找到实际脚本名
  local resolved
  resolved=( ${script} )
  if [[ ${#resolved[@]} -ne 1 || ! -f "${resolved[0]}" ]]; then
    echo "[run_batch] ERROR: cannot resolve script for exp${exp_id}" >&2
    exit 1
  fi
  echo ""
  echo "============================================================"
  echo "[run_batch] Starting exp${exp_id}: ${resolved[0]}"
  echo "[run_batch]   MAX_STEPS=${MAX_STEPS}"
  echo "[run_batch]   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
  bash "${resolved[0]}"
  echo "[run_batch] exp${exp_id} finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

# ---- 按机器号分配实验 ----
case "${MACHINE}" in
  1)
    echo "[run_batch] Machine A: exp1 → exp2 → exp3"
    run_exp 1
    run_exp 2
    run_exp 3
    ;;
  2)
    echo "[run_batch] Machine B: exp4 → exp5"
    run_exp 4
    run_exp 5
    ;;
  3)
    echo "[run_batch] Machine C: exp6 → exp7"
    run_exp 6
    run_exp 7
    ;;
esac

echo ""
echo "[run_batch] All experiments on machine ${MACHINE} completed!"
echo "[run_batch] $(date '+%Y-%m-%d %H:%M:%S')"
