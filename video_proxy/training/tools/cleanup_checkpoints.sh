#!/usr/bin/env bash
# cleanup_checkpoints.sh — 清理所有消融实验 checkpoint
#
# 递归扫描 ROOT 下所有「含 global_step_* 子目录」的实验目录：
#   - 只保留最后一个（步数最大的）global_step_*
#   - 删除所有 *.log 文件（run_*.log / summary_*.log 等）
#   - 删除 ray_logs/
#
# 用法:
#   bash local_scripts/cleanup_checkpoints.sh [ROOT]
#
# ROOT 默认为 /m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed
# 加 DRY_RUN=1 只打印，不实际删除（跳过确认）。

set -euo pipefail

ROOT="${1:-/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "$ROOT" ]]; then
    echo "[ERROR] 目录不存在: $ROOT"
    exit 1
fi

echo "=== 扫描根目录: $ROOT ==="

# ── 阶段一：扫描，收集待删列表 ────────────────────────────────
declare -a DELETE_ITEMS=()   # 每项格式: "类型|路径|大小"

total_exp=0

while IFS= read -r -d '' exp_dir; do
    exp_dir="$(dirname "$exp_dir")"

    echo ""
    echo "--- $(realpath --relative-to="$ROOT" "$exp_dir") ---"

    # 1. *.log
    while IFS= read -r -d '' f; do
        sz=$(du -sh "$f" 2>/dev/null | cut -f1)
        echo "  [log] $(basename "$f")  ($sz)"
        DELETE_ITEMS+=("log|$f|$sz")
    done < <(find "$exp_dir" -maxdepth 1 -name "*.log" -print0 2>/dev/null)

    # 2. ray_logs/
    if [[ -d "${exp_dir}/ray_logs" ]]; then
        sz=$(du -sh "${exp_dir}/ray_logs" 2>/dev/null | cut -f1)
        echo "  [ray] ray_logs/  ($sz)"
        DELETE_ITEMS+=("dir|${exp_dir}/ray_logs|$sz")
    fi

    # 3. 中间 global_step_*
    mapfile -t steps < <(find "$exp_dir" -maxdepth 1 -type d -name "global_step_*" 2>/dev/null \
        | sort -t_ -k3 -n)

    if [[ ${#steps[@]} -eq 0 ]]; then
        echo "  [ckpt] 无 global_step_* 目录"
    elif [[ ${#steps[@]} -eq 1 ]]; then
        echo "  [ckpt] 仅一个 checkpoint，跳过: $(basename "${steps[0]}")"
    else
        echo "  [ckpt] 保留: $(basename "${steps[-1]}")"
        for step_dir in "${steps[@]::${#steps[@]}-1}"; do
            sz=$(du -sh "$step_dir" 2>/dev/null | cut -f1)
            echo "  [ckpt] 删除: $(basename "$step_dir")  ($sz)"
            DELETE_ITEMS+=("dir|$step_dir|$sz")
        done
    fi

    ((total_exp++)) || true

done < <(find "$ROOT" -mindepth 2 -maxdepth 6 -type d -name "global_step_*" \
    ! -path "*/global_step_*/*" -print0 2>/dev/null | sort -z -u -t/ | \
    awk 'BEGIN{RS="\0"; ORS="\0"} {dir=$0; sub(/\/global_step_[^\/]*$/, "", dir); if (!seen[dir]++) print $0}')

echo ""
echo "=== 扫描完成，共 $total_exp 个实验目录，${#DELETE_ITEMS[@]} 项待删除 ==="

if [[ ${#DELETE_ITEMS[@]} -eq 0 ]]; then
    echo "无需清理，退出。"
    exit 0
fi

# ── 阶段二：确认 / DRY_RUN ────────────────────────────────────
if [[ "$DRY_RUN" == "1" ]]; then
    echo "(DRY_RUN 模式，不实际删除)"
    exit 0
fi

echo ""
read -r -p "确认删除以上 ${#DELETE_ITEMS[@]} 项？[y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "已取消。"
    exit 0
fi

# ── 阶段三：执行删除 ──────────────────────────────────────────
echo ""
echo "=== 开始删除 ==="
deleted=0
for item in "${DELETE_ITEMS[@]}"; do
    IFS='|' read -r _type path _sz <<< "$item"
    if rm -rf "$path"; then
        echo "  删除: $path"
        ((deleted++)) || true
    else
        echo "  [ERROR] 删除失败: $path"
    fi
done

echo ""
echo "=== 完成，成功删除 $deleted / ${#DELETE_ITEMS[@]} 项 ==="
