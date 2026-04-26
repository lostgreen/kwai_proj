#!/usr/bin/env bash
# run_fulldata_batch.sh — 连续跑全量 AOT 实验
#
# 两台机器各跑 2 个实验:
#   Machine 1: v2t_3way → v2t_binary
#   Machine 2: t2v_3way → t2v_binary
#
# 用法:
#   # 机器 A (log 自动保存到 checkpoint 目录)
#   nohup bash /m2v_intern/xuboshen/zgw/EasyR1/local_scripts/aot_ablations/run_fulldata_batch.sh 1 &
#
#   # 机器 B
#   nohup bash /m2v_intern/xuboshen/zgw/EasyR1/local_scripts/aot_ablations/run_fulldata_batch.sh 2 &
set -uo pipefail

GROUP="${1:-}"
if [ -z "$GROUP" ] || { [ "$GROUP" != "1" ] && [ "$GROUP" != "2" ]; }; then
    echo "Usage: bash run_fulldata_batch.sh <1|2>"
    echo "  1 = v2t_3way + v2t_binary (Machine A)"
    echo "  2 = t2v_3way + t2v_binary (Machine B)"
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Log 输出到 checkpoint 目录（当前目录可能只读）
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/youcook2_seg_aot/ablations_lr5e-7_kl0p04}"
LOG_DIR="${CHECKPOINT_ROOT}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/batch_group${GROUP}_$(date '+%Y%m%d_%H%M%S').log"

# Redirect all output to log file AND terminal
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Log file: ${LOG_FILE}"

run_exp() {
    local script="$1"
    local name
    name=$(basename "${script}" .sh)
    echo ""
    echo "============================================================"
    echo " [$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name}"
    echo "============================================================"

    bash "${script}" || {
        echo " [FAILED] ${name} at $(date '+%Y-%m-%d %H:%M:%S') — continuing to next" >&2
        return 0
    }

    echo " [DONE]   ${name} at $(date '+%Y-%m-%d %H:%M:%S')"
}

echo "============================================================"
echo " Full-data AOT batch — Group ${GROUP}"
echo " Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

case "${GROUP}" in
    1)
        run_exp "${SCRIPT_DIR}/exp_v2t_3way.sh"
        run_exp "${SCRIPT_DIR}/exp_v2t_binary.sh"
        ;;
    2)
        run_exp "${SCRIPT_DIR}/exp_t2v_3way.sh"
        run_exp "${SCRIPT_DIR}/exp_t2v_binary.sh"
        ;;
esac

echo ""
echo "============================================================"
echo " Group ${GROUP} done at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
