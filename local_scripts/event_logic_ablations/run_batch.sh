#!/usr/bin/env bash
# =============================================================
# run_batch.sh — Event Logic 消融实验批量运行器
#
# 用法: MAX_STEPS=60 bash local_scripts/event_logic_ablations/run_batch.sh <group>
#   group=1 → Machine A: exp1 → exp4
#   group=2 → Machine B: exp2 → exp5
#   group=3 → Machine C: exp3 → exp6
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GROUP="${1:?Usage: $0 <1|2|3>}"

run_exp() {
  local script="$1"
  local name
  name="$(basename "${script}" .sh)"
  echo ""
  echo "========================================================"
  echo " Starting ${name} at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "========================================================"
  bash "${script}" || {
    echo "[FAILED] ${name} — continuing to next experiment" >&2
    return 0
  }
  echo "[DONE] ${name} at $(date '+%Y-%m-%d %H:%M:%S')"
}

case "${GROUP}" in
  1)
    run_exp "${SCRIPT_DIR}/exp1_add_only.sh"
    run_exp "${SCRIPT_DIR}/exp4_add_replace.sh"
    ;;
  2)
    run_exp "${SCRIPT_DIR}/exp2_replace_only.sh"
    run_exp "${SCRIPT_DIR}/exp5_all_mixed.sh"
    ;;
  3)
    run_exp "${SCRIPT_DIR}/exp3_sort_only.sh"
    run_exp "${SCRIPT_DIR}/exp6_all_filtered.sh"
    ;;
  *)
    echo "Unknown group: ${GROUP}. Use 1, 2, or 3." >&2
    exit 1
    ;;
esac

echo ""
echo "[run_batch] Group ${GROUP} finished at $(date '+%Y-%m-%d %H:%M:%S')"
