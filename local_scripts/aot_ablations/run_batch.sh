#!/usr/bin/env bash
# =============================================================
# run_batch.sh — 一键运行 AoT 消融实验（3 台机器各跑 3 个实验）
#
# 用法:
#   bash local_scripts/aot_ablations/run_batch.sh 1   # 机器 A: exp1 → exp2 → exp7
#   bash local_scripts/aot_ablations/run_batch.sh 2   # 机器 B: exp3 → exp4 → exp9
#   bash local_scripts/aot_ablations/run_batch.sh 3   # 机器 C: exp5 → exp6 → exp8
#
# 分配逻辑（按数据依赖排列）:
#   机器 A: exp1, exp2 先跑（产出 report），然后 exp7（依赖 exp1+exp2）
#   机器 B: exp3, exp4 先跑（产出 report），然后 exp9（依赖 exp3+exp4）
#   机器 C: exp5, exp6 先跑（无外部依赖），然后 exp8（依赖 exp1-4，等 A/B 产出后启动）
#
# 可覆盖:
#   MAX_STEPS=60  TASK_HOMOGENEOUS=false  bash local_scripts/aot_ablations/run_batch.sh 1
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---- 默认参数（快速迭代模式）----
export MAX_STEPS="${MAX_STEPS:-60}"
export TASK_HOMOGENEOUS="${TASK_HOMOGENEOUS:-false}"

MACHINE="${1:-}"
if [[ -z "${MACHINE}" || ! "${MACHINE}" =~ ^[1-3]$ ]]; then
  echo "Usage: bash $0 <machine_id:1|2|3>"
  echo ""
  echo "  1 → exp1, exp2, exp7   (机器 A)"
  echo "  2 → exp3, exp4, exp9   (机器 B)"
  echo "  3 → exp5, exp6, exp8   (机器 C)"
  exit 1
fi

# ---- 等待文件就绪（用于跨机器依赖） ----
wait_for_files() {
  local timeout="${WAIT_TIMEOUT:-7200}"  # 默认最多等 2 小时
  local interval=30
  local elapsed=0
  local files=("$@")

  for f in "${files[@]}"; do
    while [[ ! -f "${f}" ]]; do
      if (( elapsed >= timeout )); then
        echo "[run_batch] TIMEOUT: waited ${timeout}s for ${f}" >&2
        exit 1
      fi
      echo "[run_batch] Waiting for: ${f}  (${elapsed}/${timeout}s)"
      sleep "${interval}"
      elapsed=$(( elapsed + interval ))
    done
    echo "[run_batch] Found: ${f}"
  done
}

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
  echo "[run_batch]   MAX_STEPS=${MAX_STEPS}  TASK_HOMOGENEOUS=${TASK_HOMOGENEOUS}"
  echo "[run_batch]   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"
  bash "${resolved[0]}"
  echo "[run_batch] exp${exp_id} finished at $(date '+%Y-%m-%d %H:%M:%S')"
}

# ---- AOT_DATA_ROOT（需要和 common.sh 一致）----
source "${SCRIPT_DIR}/common.sh"
_ABL="${AOT_DATA_ROOT}/ablations_refined"

# ---- 按机器号分配实验 ----
case "${MACHINE}" in
  1)
    echo "[run_batch] Machine A: exp1 → exp2 → exp7"
    run_exp 1
    run_exp 2
    run_exp 7
    ;;
  2)
    echo "[run_batch] Machine B: exp3 → exp4 → exp9"
    run_exp 3
    run_exp 4
    run_exp 9
    ;;
  3)
    echo "[run_batch] Machine C: exp5 → exp6 → exp8"
    run_exp 5
    run_exp 6
    # exp8 依赖 exp1-4 的 report，可能由机器 A/B 生成，需要等
    echo "[run_batch] Checking exp8 prerequisites (exp1-4 reports) ..."
    wait_for_files \
      "${_ABL}/exp1/offline_filter_report.jsonl" \
      "${_ABL}/exp2/offline_filter_report.jsonl" \
      "${_ABL}/exp3/offline_filter_report.jsonl" \
      "${_ABL}/exp4/offline_filter_report.jsonl"
    run_exp 8
    ;;
esac

echo ""
echo "[run_batch] All experiments on machine ${MACHINE} completed!"
echo "[run_batch] $(date '+%Y-%m-%d %H:%M:%S')"
