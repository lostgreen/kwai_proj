#!/usr/bin/env bash
# ── LLaVA-Video-178K MCQ: Pilot Rollout → Filter → Downsample Pipeline ──
#
# 流程:
#   Step 1: prepare_mcq.py — 解析 MCQ JSON → 统一 JSONL
#   Step 2: sample_pilot.py — 分层采样 pilot set (~500/cell)
#   Step 3: offline_rollout_filter.py — rollout + reward 计算
#   Step 4: filter_and_downsample.py — 按准确率过滤 + 均匀下采样
#   Step 5: visualize.py — 分布对比图 (before vs after)
#
# 用法 (从 train/ 目录运行):
#   bash proxy_data/llava_video_178k/run_pipeline.sh
#
# 环境变量:
#   DATASET_ROOT  — LLaVA-Video-178K 数据集根目录
#   MODEL_PATH    — 模型路径 (默认 Qwen3-VL-4B-Instruct)
#   OUTPUT_ROOT   — 输出目录
#   NUM_GPUS      — GPU 数 (默认 2)
#   TP_SIZE       — tensor parallel size (默认 1; TP>1 时用 TP 模式)
#   NUM_ROLLOUTS  — rollout 次数 (默认 4)
#   PER_CELL      — 每个 (时长×来源) 格子采样数 (默认 500)
#   TARGET_TOTAL  — 最终下采样目标条数 (默认 1000)
#   MIN_ACC       — 最低准确率阈值 (默认 0.25)
#   MAX_ACC       — 最高准确率阈值 (默认 0.5)
#   BATCH_SIZE    — vLLM batch size (默认 16)
#   FORCE         — 强制重跑 (1 = 强制, 默认 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ── 配置 ──
DATASET_ROOT="${DATASET_ROOT:-/ytech_m2v5_hdd/workspace/kling_mm/Datasets/LLaVA-Video-178K}"
MODEL_PATH="${MODEL_PATH:-/home/xuboshen/models/Qwen3-VL-4B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/results}"
NUM_GPUS="${NUM_GPUS:-2}"
TP_SIZE="${TP_SIZE:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
PER_CELL="${PER_CELL:-5000}"
TARGET_TOTAL="${TARGET_TOTAL:-1000}"
MIN_ACC="${MIN_ACC:-0.25}"
MAX_ACC="${MAX_ACC:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"

REWARD_FN="$SCRIPT_DIR/mcq_reward.py:compute_score"
ROLLOUT_SCRIPT="$REPO_ROOT/local_scripts/offline_rollout_filter.py"

echo "============================================="
echo " LLaVA-Video-178K MCQ Pipeline"
echo " Dataset:    $DATASET_ROOT"
echo " Model:      $MODEL_PATH"
echo " Output:     $OUTPUT_ROOT"
echo " GPUs:       $NUM_GPUS (TP=$TP_SIZE)"
echo " Rollouts:   $NUM_ROLLOUTS"
echo " Per Cell:   $PER_CELL"
echo " Acc Range:  ($MIN_ACC, $MAX_ACC)"
echo " Target:     $TARGET_TOTAL"
echo "============================================="

mkdir -p "$OUTPUT_ROOT"

# ── Step 1: 解析 MCQ ──
MCQ_JSONL="$OUTPUT_ROOT/mcq_all.jsonl"
if [ "$FORCE" != "1" ] && [ -s "$MCQ_JSONL" ]; then
    COUNT=$(wc -l < "$MCQ_JSONL" | tr -d ' ')
    echo ""
    echo "=== Step 1: prepare_mcq [已完成: $COUNT records — 跳过] ==="
else
    echo ""
    echo "=== Step 1: prepare_mcq ==="
    python "$SCRIPT_DIR/prepare_mcq.py" \
        --dataset-root "$DATASET_ROOT" \
        --output "$MCQ_JSONL"
fi

# ── Step 2: 分层采样 ──
PILOT_JSONL="$OUTPUT_ROOT/pilot_sample.jsonl"
if [ "$FORCE" != "1" ] && [ -s "$PILOT_JSONL" ]; then
    COUNT=$(wc -l < "$PILOT_JSONL" | tr -d ' ')
    echo ""
    echo "=== Step 2: sample_pilot [已完成: $COUNT records — 跳过] ==="
else
    echo ""
    echo "=== Step 2: sample_pilot ==="
    python "$SCRIPT_DIR/sample_pilot.py" \
        --input "$MCQ_JSONL" \
        --output "$PILOT_JSONL" \
        --per-cell "$PER_CELL" \
        --seed "$SEED"
fi

# ── Step 3: Rollout (with resume support) ──
ROLLOUT_OUTPUT="$OUTPUT_ROOT/rollout_kept.jsonl"
ROLLOUT_REPORT="$OUTPUT_ROOT/rollout_report.jsonl"
ROLLOUT_DONE=0
RESUME_INPUT=""

if [ "$FORCE" != "1" ] && [ -s "$ROLLOUT_REPORT" ]; then
    REPORT_COUNT=$(wc -l < "$ROLLOUT_REPORT" | tr -d ' ')
    PILOT_COUNT=$(wc -l < "$PILOT_JSONL" | tr -d ' ')
    if [ "$REPORT_COUNT" -ge "$PILOT_COUNT" ]; then
        echo ""
        echo "=== Step 3: rollout [已完成: $REPORT_COUNT/$PILOT_COUNT — 跳过] ==="
        ROLLOUT_DONE=1
    else
        # Partial progress — resume mode
        echo ""
        echo "=== Step 3: rollout [断点续传: 已完成 $REPORT_COUNT/$PILOT_COUNT] ==="
        RESUME_INPUT="$OUTPUT_ROOT/_remaining.jsonl"

        python "$SCRIPT_DIR/resume_helper.py" \
            --input "$PILOT_JSONL" \
            --report "$ROLLOUT_REPORT" \
            --output "$RESUME_INPUT" || {
                RESUME_EXIT=$?
                if [ "$RESUME_EXIT" = "42" ]; then
                    echo "  All items already processed!"
                    ROLLOUT_DONE=1
                else
                    echo "  Resume helper failed (exit=$RESUME_EXIT)"
                    exit 1
                fi
            }
    fi
fi

if [ "$ROLLOUT_DONE" = "0" ]; then
    # Determine which input to use
    EFFECTIVE_INPUT="${RESUME_INPUT:-$PILOT_JSONL}"
    REMAINING_COUNT=$(wc -l < "$EFFECTIVE_INPUT" | tr -d ' ')
    echo ""
    echo "=== Step 3: rollout (${REMAINING_COUNT} items × rollout_n=$NUM_ROLLOUTS) ==="

    # For resume: write to temp files, merge afterwards
    if [ -n "$RESUME_INPUT" ]; then
        RUN_OUTPUT="$OUTPUT_ROOT/_resume_kept.jsonl"
        RUN_REPORT="$OUTPUT_ROOT/_resume_report.jsonl"
    else
        RUN_OUTPUT="$ROLLOUT_OUTPUT"
        RUN_REPORT="$ROLLOUT_REPORT"
    fi

    ROLLOUT_COMMON=(
        --input_jsonl "$EFFECTIVE_INPUT"
        --output_jsonl "$RUN_OUTPUT"
        --report_jsonl "$RUN_REPORT"
        --model_path "$MODEL_PATH"
        --reward_function "$REWARD_FN"
        --backend vllm
        --num_rollouts "$NUM_ROLLOUTS"
        --temperature 0.7
        --max_new_tokens 256
        --gpu_memory_utilization "$GPU_MEM_UTIL"
        --max_num_batched_tokens "$MAX_BATCHED_TOKENS"
        --min_mean_reward 0.0
        --max_mean_reward 1.0
        --seed "$SEED"
    )

    if [ "$TP_SIZE" -gt 1 ]; then
        echo "  TP=$TP_SIZE mode"
        python "$ROLLOUT_SCRIPT" \
            "${ROLLOUT_COMMON[@]}" \
            --tensor_parallel_size "$TP_SIZE"
    elif [ "$NUM_GPUS" -gt 1 ]; then
        echo "  Data-parallel mode (${NUM_GPUS} GPUs)"
        for i in $(seq 0 $((NUM_GPUS-1))); do
            CUDA_VISIBLE_DEVICES=$i python "$ROLLOUT_SCRIPT" \
                "${ROLLOUT_COMMON[@]}" \
                --tensor_parallel_size 1 \
                --shard_id "$i" --num_shards "$NUM_GPUS" \
                --output_jsonl "$OUTPUT_ROOT/_shard${i}_kept.jsonl" \
                --report_jsonl "$OUTPUT_ROOT/_shard${i}_report.jsonl" &
        done
        wait
        cat "$OUTPUT_ROOT"/_shard*_kept.jsonl > "$RUN_OUTPUT"
        cat "$OUTPUT_ROOT"/_shard*_report.jsonl > "$RUN_REPORT"
        rm -f "$OUTPUT_ROOT"/_shard*_kept.jsonl "$OUTPUT_ROOT"/_shard*_report.jsonl
    else
        echo "  Single GPU mode"
        python "$ROLLOUT_SCRIPT" \
            "${ROLLOUT_COMMON[@]}" \
            --tensor_parallel_size 1
    fi

    # Merge resume results with previous run
    if [ -n "$RESUME_INPUT" ] && [ -s "$RUN_REPORT" ]; then
        echo "  Merging resume results..."
        cat "$RUN_OUTPUT" >> "$ROLLOUT_OUTPUT"
        cat "$RUN_REPORT" >> "$ROLLOUT_REPORT"
        rm -f "$RESUME_INPUT" "$RUN_OUTPUT" "$RUN_REPORT"
        MERGED_COUNT=$(wc -l < "$ROLLOUT_REPORT" | tr -d ' ')
        echo "  Merged report: $MERGED_COUNT total records"
    fi
fi

# ── Step 4: 过滤 + 下采样 ──
FINAL_JSONL="$OUTPUT_ROOT/train_final.jsonl"
echo ""
echo "=== Step 4: filter_and_downsample ==="
python "$SCRIPT_DIR/filter_and_downsample.py" \
    --report "$ROLLOUT_REPORT" \
    --input "$PILOT_JSONL" \
    --output "$FINAL_JSONL" \
    --min-acc "$MIN_ACC" \
    --max-acc "$MAX_ACC" \
    --target-total "$TARGET_TOTAL" \
    --seed "$SEED"

# ── Summary ──
echo ""
echo "=========================================="
echo " Pipeline done!"
if [ -s "$FINAL_JSONL" ]; then
    FINAL_COUNT=$(wc -l < "$FINAL_JSONL" | tr -d ' ')
    echo " Final:  $FINAL_JSONL ($FINAL_COUNT records)"
fi
echo "=========================================="

# ── Step 5: 可视化 ──
FIGURES_DIR="$OUTPUT_ROOT/figures"
echo ""
echo "=== Step 5: visualize ==="
VIS_ARGS=(
    --before "$MCQ_JSONL"
    --after "$FINAL_JSONL"
    --outdir "$FIGURES_DIR"
)
if [ -s "$ROLLOUT_REPORT" ]; then
    VIS_ARGS+=(--report "$ROLLOUT_REPORT")
fi
python "$SCRIPT_DIR/visualize.py" "${VIS_ARGS[@]}"
echo ""
echo " Figures: $FIGURES_DIR/"
