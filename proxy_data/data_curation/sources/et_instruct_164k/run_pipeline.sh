#!/usr/bin/env bash
# ── ET-Instruct-164K: 两阶段 LLM 筛选 pipeline ──
#
# 前提: 已运行 text_filter.py 产出 results/passed.jsonl
#
# 用法:
#   # 抽样试跑 (Stage A 200条)
#   bash run_pipeline.sh --sample
#
#   # 全量运行 (Stage A + B, 断点续评)
#   bash run_pipeline.sh --full
#
#   # 仅 Stage B (Stage A 已完成)
#   bash run_pipeline.sh --stage-b-only

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 配置 ──
API_BASE="${API_BASE:-https://api.novita.ai/v3/openai}"
MODEL="${MODEL:-pa/gmn-2.5-pr}"
WORKERS="${WORKERS:-8}"
INPUT="${INPUT:-results/passed.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# ── 参数解析 ──
MODE="${1:---sample}"

echo "============================================="
echo " ET-Instruct-164K: 两阶段 LLM 筛选"
echo " Mode: $MODE"
echo " API: $API_BASE"
echo " Model: $MODEL"
echo " Workers: $WORKERS"
echo " Input: $INPUT"
echo "============================================="

if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT 不存在"
    echo "请先运行 text_filter.py 产出 passed.jsonl:"
    echo "  python text_filter.py \\"
    echo "      --json_path /path/to/et_instruct_164k_txt.json \\"
    echo "      --output_dir results \\"
    echo "      --config ../../configs/et_instruct_164k.yaml"
    exit 1
fi

# ── Stage A: L2 粒度粗筛 ──
run_stage_a() {
    local sample_args=""
    if [[ "$MODE" == "--sample" ]]; then
        sample_args="--sample-n 200"
        echo ""
        echo "[Stage A] 抽样 200 条粗筛..."
    else
        sample_args="--no-sample --resume"
        echo ""
        echo "[Stage A] 全量粗筛 (断点续评)..."
    fi

    python stage_a_coarse_filter.py \
        --input "$INPUT" \
        --output "$RESULTS_DIR/stage_a_results.jsonl" \
        --api-base "$API_BASE" \
        --model "$MODEL" \
        --workers "$WORKERS" \
        $sample_args

    echo ""
    echo "[Stage A] 完成。查看结果:"
    echo "  keep:   $RESULTS_DIR/stage_a_results_keep.jsonl"
    echo "  maybe:  $RESULTS_DIR/stage_a_results_maybe.jsonl"
    echo "  reject: $RESULTS_DIR/stage_a_results_reject.jsonl"
}

# ── 可选: 用硬规则校正 Stage A decision ──
run_stage_a_rules() {
    if [ -f "$RESULTS_DIR/stage_a_results.jsonl" ]; then
        echo ""
        echo "[Rules] 应用 Stage A 程序化决策规则..."
        python ../shared/decision_rules.py \
            --input "$RESULTS_DIR/stage_a_results.jsonl" \
            --output "$RESULTS_DIR/stage_a_ruled.jsonl" \
            --stage A --override
    fi
}

# ── Stage B: 层次潜力精筛 ──
run_stage_b() {
    local stage_a_keep="$RESULTS_DIR/stage_a_results_keep.jsonl"
    if [ ! -f "$stage_a_keep" ]; then
        echo "Error: $stage_a_keep 不存在，请先运行 Stage A"
        exit 1
    fi

    local keep_count
    keep_count=$(wc -l < "$stage_a_keep" | tr -d ' ')
    echo ""
    echo "[Stage B] 精筛 $keep_count 条 Stage A keep 样本..."

    python ../shared/stage_b_fine_filter.py \
        --input "$stage_a_keep" \
        --output "$RESULTS_DIR/stage_b_results.jsonl" \
        --data-source et_instruct \
        --api-base "$API_BASE" \
        --model "$MODEL" \
        --workers "$WORKERS" \
        --no-sample --resume

    echo ""
    echo "[Stage B] 完成。最终结果:"
    echo "  keep:   $RESULTS_DIR/stage_b_results_keep.jsonl"
    echo "  maybe:  $RESULTS_DIR/stage_b_results_maybe.jsonl"
    echo "  reject: $RESULTS_DIR/stage_b_results_reject.jsonl"
}

# ── 执行 ──
case "$MODE" in
    --sample)
        run_stage_a
        echo ""
        echo "抽样完成。检查 Stage A 分布后，用 --full 运行全量。"
        ;;
    --full)
        run_stage_a
        run_stage_a_rules
        run_stage_b
        echo ""
        echo "========== Pipeline 完成 =========="
        if [ -f "$RESULTS_DIR/stage_b_results_keep.jsonl" ]; then
            final_count=$(wc -l < "$RESULTS_DIR/stage_b_results_keep.jsonl" | tr -d ' ')
            echo "最终保留: $final_count 条样本"
        fi
        ;;
    --stage-b-only)
        run_stage_b
        ;;
    *)
        echo "用法: bash run_pipeline.sh [--sample|--full|--stage-b-only]"
        exit 1
        ;;
esac
