#!/usr/bin/env bash
# ── LLaVA-Video-178K MCQ: Pilot Rollout → Filter → Downsample Pipeline ──
#
# 流程:
#   Step 1: prepare_mcq.py — 解析 MCQ JSON → 统一 JSONL
#   Step 2: sample_pilot.py — 分层采样 pilot set (pilot 模式)
#   Step 3: offline_rollout_filter.py — rollout + reward 计算
#   Step 4: filter_and_downsample.py — 按准确率过滤 + 均匀下采样
#   Step 5: visualize.py — 分布对比图 (before vs after)
#
# 用法 (从 train/ 目录运行):
#   bash proxy_data/llava_video_178k/run_pipeline.sh
#
# 环境变量:
#   DATASET_ROOT  — LLaVA-Video-178K 数据集根目录
#   MODEL_PATH    — 模型路径 (默认 Qwen3-VL-8B-Instruct)
#   OUTPUT_ROOT   — 输出目录
#   NUM_GPUS      — GPU 数 (默认 8)
#   TP_SIZE       — tensor parallel size (默认 1; TP>1 时用 TP 模式)
#   NUM_ROLLOUTS  — rollout 次数 (默认 8)
#   ROLLOUT_SCOPE — pilot 或 full (默认 full)
#   PER_CELL      — 每个 (时长×来源) 格子采样数 (默认 5000; 仅 pilot 模式)
#   TARGET_TOTAL  — 最终下采样目标条数 (默认 pilot=1000, full=0; 0=不过采样)
#   MIN_ACC       — 最低准确率阈值 (默认 0.0)
#   MAX_ACC       — 最高准确率阈值 (默认 0.375 = 3/8)
#   BATCH_SIZE    — vLLM batch size (默认 16)
#   ENABLE_GPU_FILLER / FILLER_START_DELAY / FILLER_TARGET_UTIL
#                 — rollout 阶段的 GPU filler 控制项
#   FORCE         — 强制重跑 (1 = 强制, 默认 0)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "${REPO_ROOT}/local_scripts/gpu_filler_common.sh"

# ── 配置 ──
# 默认直接对齐当前这次实验:
#   Qwen3-VL-8B + 8 GPU + full rollout + rollout_n=8 + 保留 mean_acc <= 3/8
#   filler 采用当前版本的 signal 模式，默认参数偏向把 avg util 顶到 70+ 附近
DATASET_ROOT="${DATASET_ROOT:-/ytech_m2v5_hdd/workspace/kling_mm/Datasets/LLaVA-Video-178K}"
MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/results_qwen3_vl_8b_roll8_leq3of8}"
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
ROLLOUT_SCOPE="${ROLLOUT_SCOPE:-full}"
PER_CELL="${PER_CELL:-5000}"
if [[ -n "${TARGET_TOTAL:-}" ]]; then
    TARGET_TOTAL="${TARGET_TOTAL}"
elif [[ "${ROLLOUT_SCOPE}" == "full" ]]; then
    TARGET_TOTAL="0"
else
    TARGET_TOTAL="1000"
fi
MIN_ACC="${MIN_ACC:-0.0}"
MAX_ACC="${MAX_ACC:-0.375}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-24576}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.82}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
ENABLE_GPU_FILLER="${ENABLE_GPU_FILLER:-true}"
FILLER_LOG_PATH="${FILLER_LOG_PATH:-$OUTPUT_ROOT/gpu_filler.log}"
FILLER_START_DELAY="${FILLER_START_DELAY:-0}"
FILLER_MODE="${FILLER_MODE:-signal}"
FILLER_PER_GPU="${FILLER_PER_GPU:-true}"
FILLER_SIGNAL_PREFIX="${FILLER_SIGNAL_PREFIX:-/tmp/llava_video_gpu_phase_gpu}"
FILLER_TARGET_UTIL="${FILLER_TARGET_UTIL:-80}"
FILLER_BATCH="${FILLER_BATCH:-50}"
FILLER_MATRIX="${FILLER_MATRIX:-8192}"
FILLER_GAP_MATRIX="${FILLER_GAP_MATRIX:-4096}"
FILLER_PUSH_MATRIX="${FILLER_PUSH_MATRIX:-6144}"
FILLER_BUSY_MATRIX="${FILLER_BUSY_MATRIX:-3072}"
FILLER_BUSY_BATCH="${FILLER_BUSY_BATCH:-8}"
FILLER_BUSY_SLEEP_MS="${FILLER_BUSY_SLEEP_MS:-10}"
FILLER_IDLE_SLEEP_MS="${FILLER_IDLE_SLEEP_MS:-6}"
FILLER_ORPHAN_MATRIX="${FILLER_ORPHAN_MATRIX:-4096}"
FILLER_ORPHAN_BATCH="${FILLER_ORPHAN_BATCH:-16}"
FILLER_ORPHAN_SLEEP_MS="${FILLER_ORPHAN_SLEEP_MS:-8}"
FILLER_BUSY_HOLD_MS="${FILLER_BUSY_HOLD_MS:-1600}"
STOP_GPU_FILLER_ON_EXIT="${STOP_GPU_FILLER_ON_EXIT:-false}"

REWARD_FN="$SCRIPT_DIR/mcq_reward.py:compute_score"
ROLLOUT_SCRIPT="$REPO_ROOT/local_scripts/offline_rollout_filter.py"

case "${ROLLOUT_SCOPE}" in
    pilot|full) ;;
    *)
        echo "Unsupported ROLLOUT_SCOPE=${ROLLOUT_SCOPE} (expected pilot or full)" >&2
        exit 1
        ;;
esac

VISIBLE_GPU_TOKENS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a VISIBLE_GPU_TOKENS <<< "${CUDA_VISIBLE_DEVICES}"
else
    for ((i=0; i<NUM_GPUS; i++)); do
        VISIBLE_GPU_TOKENS+=("${i}")
    done
fi
if (( NUM_GPUS > ${#VISIBLE_GPU_TOKENS[@]} )); then
    echo "NUM_GPUS=${NUM_GPUS}, but only ${#VISIBLE_GPU_TOKENS[@]} visible GPUs are available" >&2
    exit 1
fi

if [[ "${TP_SIZE}" -gt 1 && "${FILLER_PER_GPU}" =~ ^(true|TRUE|1|yes|YES)$ ]]; then
    echo "TP mode detected; disabling per-GPU filler to keep a single shared phase signal."
    FILLER_PER_GPU=false
fi

if [[ -z "${FILLER_GPUS:-}" && "${NUM_GPUS}" -gt 0 ]]; then
    if [[ "${FILLER_PER_GPU}" =~ ^(true|TRUE|1|yes|YES)$ ]]; then
        FILLER_GPUS="$(printf '%s,' "${VISIBLE_GPU_TOKENS[@]:0:${NUM_GPUS}}")"
        FILLER_GPUS="${FILLER_GPUS%,}"
    else
        FILLER_GPUS="$(seq -s, 0 $((NUM_GPUS-1)))"
    fi
fi

echo "============================================="
echo " LLaVA-Video-178K MCQ Pipeline"
echo " Dataset:    $DATASET_ROOT"
echo " Model:      $MODEL_PATH"
echo " Output:     $OUTPUT_ROOT"
echo " GPUs:       $NUM_GPUS (TP=$TP_SIZE)"
echo " Rollouts:   $NUM_ROLLOUTS"
echo " Scope:      $ROLLOUT_SCOPE"
echo " Per Cell:   $PER_CELL"
echo " Acc Range:  [$MIN_ACC, $MAX_ACC]"
echo " Target:     $TARGET_TOTAL"
echo " Batch:      $BATCH_SIZE  Max Tokens: $MAX_BATCHED_TOKENS"
echo " Filler:     enabled=$ENABLE_GPU_FILLER mode=$FILLER_MODE per_gpu=$FILLER_PER_GPU target=$FILLER_TARGET_UTIL"
echo "============================================="

mkdir -p "$OUTPUT_ROOT"
trap 'gpu_filler_cleanup' EXIT

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
ROLLOUT_SOURCE_JSONL="$MCQ_JSONL"
if [[ "$ROLLOUT_SCOPE" == "pilot" ]]; then
    ROLLOUT_SOURCE_JSONL="$PILOT_JSONL"
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
else
    FULL_COUNT=$(wc -l < "$MCQ_JSONL" | tr -d ' ')
    echo ""
    echo "=== Step 2: sample_pilot [full mode — 跳过, 直接 rollout 全量 $FULL_COUNT records] ==="
fi

# ── Step 3: Rollout (with resume support) ──
ROLLOUT_OUTPUT="$OUTPUT_ROOT/rollout_kept.jsonl"
ROLLOUT_REPORT="$OUTPUT_ROOT/rollout_report.jsonl"
ROLLOUT_DONE=0
RESUME_INPUT=""

if [ "$FORCE" != "1" ] && [ -s "$ROLLOUT_REPORT" ]; then
    REPORT_COUNT=$(wc -l < "$ROLLOUT_REPORT" | tr -d ' ')
    SOURCE_COUNT=$(wc -l < "$ROLLOUT_SOURCE_JSONL" | tr -d ' ')
    if [ "$REPORT_COUNT" -ge "$SOURCE_COUNT" ]; then
        echo ""
        echo "=== Step 3: rollout [已完成: $REPORT_COUNT/$SOURCE_COUNT — 跳过] ==="
        ROLLOUT_DONE=1
    else
        # Partial progress — resume mode
        echo ""
        echo "=== Step 3: rollout [断点续传: 已完成 $REPORT_COUNT/$SOURCE_COUNT] ==="
        RESUME_INPUT="$OUTPUT_ROOT/_remaining.jsonl"

        python "$SCRIPT_DIR/resume_helper.py" \
            --input "$ROLLOUT_SOURCE_JSONL" \
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
    EFFECTIVE_INPUT="${RESUME_INPUT:-$ROLLOUT_SOURCE_JSONL}"
    REMAINING_COUNT=$(wc -l < "$EFFECTIVE_INPUT" | tr -d ' ')
    echo ""
    echo "=== Step 3: rollout (${REMAINING_COUNT} items × rollout_n=$NUM_ROLLOUTS) ==="

    gpu_filler_start "[llava-video]"

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
        --batch_size "$BATCH_SIZE"
        --min_mean_reward 0.0
        --max_mean_reward 1.0
        --seed "$SEED"
    )

    if [ "$TP_SIZE" -gt 1 ]; then
        echo "  TP=$TP_SIZE mode"
        VERL_GPU_SIGNAL_PATH="${FILLER_SIGNAL_PREFIX}tp" python "$ROLLOUT_SCRIPT" \
            "${ROLLOUT_COMMON[@]}" \
            --tensor_parallel_size "$TP_SIZE"
    elif [ "$NUM_GPUS" -gt 1 ]; then
        echo "  Data-parallel mode (${NUM_GPUS} GPUs)"
        for i in $(seq 0 $((NUM_GPUS-1))); do
            SHARD_GPU="${VISIBLE_GPU_TOKENS[$i]}"
            SHARD_GPU="${SHARD_GPU//[[:space:]]/}"
            SHARD_SIGNAL_PATH="${FILLER_SIGNAL_PREFIX}${SHARD_GPU}"
            echo "    shard ${i} -> CUDA_VISIBLE_DEVICES=${SHARD_GPU}"
            VERL_GPU_SIGNAL_PATH="${SHARD_SIGNAL_PATH}" CUDA_VISIBLE_DEVICES="${SHARD_GPU}" python "$ROLLOUT_SCRIPT" \
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
        VERL_GPU_SIGNAL_PATH="${FILLER_SIGNAL_PREFIX}${VISIBLE_GPU_TOKENS[0]}" python "$ROLLOUT_SCRIPT" \
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
    --input "$MCQ_JSONL" \
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
