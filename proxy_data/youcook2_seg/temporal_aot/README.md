# Temporal AoT（时序方向判断）

> **当前推荐管线**：`build_aot_from_seg.py` 直接从层次分割标注（`hier_seg_annotation/annotations/`）生成六种 AoT 任务数据，覆盖 L1/L2/L3 三层粒度，**无需独立的 VLM captioning 步骤**。
> 所有 prompt 已领域通用化，不再绑定特定数据源（如 cooking）。

---

## 文件结构

```
temporal_aot/
├── build_aot_from_seg.py       ← ★ 主入口：从 L1/L2/L3 分割标注构建 6 种 AOT 任务
├── prompts.py                  ← AOT prompt 模板库（用于旧版 VLM captioning 管线）
├── rebalance_aot_answers.py    ← 答案重平衡工具（离线过滤后使用）
├── data/                       ← 生成数据目录
└── legacy/                     ← ⚠️ 旧版管线脚本（需独立 VLM captioning，已废弃）
    ├── build_event_aot_data.py
    ├── annotate_event_captions.py
    ├── check_and_refine_captions.py
    ├── build_aot_mcq.py
    └── mix_aot_with_youcook2.py
```

---

## 整体架构

### 数据流

```
hier_seg_annotation/annotations/*.json
    │  (共享 L1/L2/L3 三层分割标注)
    │
    ├─► phase_v2t  : L1 全视频 → 判断哪个阶段列表顺序正确       (A/B/C 三选一)
    ├─► phase_t2v  : forward 阶段列表 → 从三个 L1 clip 中选匹配的 (A/B/C 三选一)
    ├─► event_v2t  : L2 window clip → 判断哪个事件列表顺序正确    (A/B/C 三选一)
    ├─► event_t2v  : forward 事件列表 → 从三个 L2 clip 中选匹配的 (A/B/C 三选一)
    ├─► action_v2t : L3 event clip → 判断哪个动作列表顺序正确     (A/B 二选一)
    └─► action_t2v : forward 动作列表 → 从两个 L3 clip 中选匹配的 (A/B 二选一)
```

直接复用标注数据：
- L1 `macro_phases[*].phase_name` 作为阶段描述
- L2 `events[*].instruction` 作为事件描述
- L3 `grounding_results[*].sub_action` 作为动作描述

### 六种任务类型（3 粒度 × 2 方向）

| 任务 | problem_type | 粒度 | 形式 |
|------|-------------|------|------|
| **phase V2T** | `seg_aot_phase_v2t` | L1 (phase) | 给 L1 全视频，判断哪个阶段列表顺序正确（A/B/C）|
| **phase T2V** | `seg_aot_phase_t2v` | L1 (phase) | 给 forward 阶段列表，从三个 L1 clip 中选匹配的（A/B/C）|
| **event V2T** | `seg_aot_event_v2t` | L2 (event) | 给 L2 window clip，判断哪个事件列表顺序正确（A/B/C）|
| **event T2V** | `seg_aot_event_t2v` | L2 (event) | 给 forward 事件列表，从三个 L2 clip 中选匹配的（A/B/C）|
| **action V2T** | `seg_aot_action_v2t` | L3 (action) | 给 L3 event clip，判断哪个动作列表顺序正确（A/B）|
| **action T2V** | `seg_aot_action_t2v` | L3 (action) | 给 forward 动作列表，从两个 L3 clip 中选匹配的（A/B）|

---

## 快速开始

### 前提条件

1. 三层标注已完成（`hier_seg_annotation/annotations/*.json`）
2. L1 clips 已由 `hier_seg_annotation/prepare_clips.py` 生成（`clips/L1/*.mp4`）
3. L2 clips 已由 `hier_seg_annotation/prepare_clips.py` 生成（`clips/L2/*.mp4`）
4. L3 clips 已由 `hier_seg_annotation/prepare_clips.py` 生成（`clips/L3/*.mp4`）

### 一键构建（全部 6 种任务）

```bash
cd /path/to/VideoProxy/train

export ANN_DIR=/path/to/hier_seg_annotation/annotations
export CLIP_L1=/path/to/hier_seg_annotation/clips/L1
export CLIP_L2=/path/to/hier_seg_annotation/clips/L2
export CLIP_L3=/path/to/hier_seg_annotation/clips/L3
export OUTPUT_DIR=proxy_data/youcook2_seg/temporal_aot/data

python proxy_data/youcook2_seg/temporal_aot/build_aot_from_seg.py \
    --annotation-dir $ANN_DIR \
    --clip-dir-l1    $CLIP_L1 \
    --clip-dir-l2    $CLIP_L2 \
    --clip-dir-l3    $CLIP_L3 \
    --output-dir     $OUTPUT_DIR \
    --tasks phase_v2t phase_t2v action_v2t action_t2v event_v2t event_t2v \
    --complete-only \
    --total-val 200 \
    --seed 42
```

### 只构建某一层任务

```bash
# 只构建 L1 阶段级任务
python proxy_data/youcook2_seg/temporal_aot/build_aot_from_seg.py \
    --annotation-dir $ANN_DIR \
    --clip-dir-l1    $CLIP_L1 \
    --output-dir     $OUTPUT_DIR/phase_only \
    --tasks phase_v2t phase_t2v \
    --complete-only

# 只构建 L2 事件级任务
python proxy_data/youcook2_seg/temporal_aot/build_aot_from_seg.py \
    --annotation-dir $ANN_DIR \
    --clip-dir-l2    $CLIP_L2 \
    --output-dir     $OUTPUT_DIR/event_only \
    --tasks event_v2t event_t2v \
    --complete-only
```

### 输出结构

```
$OUTPUT_DIR/
├── train.jsonl   # 打乱混合的训练数据
├── val.jsonl     # 验证集
└── stats.json    # 各 problem_type 数量统计
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tasks` | 全部 6 种 | 要构建的任务，可任意组合 |
| `--clip-dir-l1` | 空 | L1 clips 目录，留空则用原始视频路径 |
| `--clip-dir-l2` | 空 | L2 clips 目录，留空则用原始视频路径 |
| `--clip-dir-l3` | 空 | L3 clips 目录，留空则用原始视频路径 |
| `--l1-fps` | 1 | L1 视频帧率 |
| `--l2-window-size` | 128 | L2 滑窗大小（秒），`-1` = 不切窗 |
| `--l2-stride` | 64 | L2 滑窗步长（秒） |
| `--min-phases` | 3 | phase_* 任务每视频最少 phase 数 |
| `--min-events` | 3 | event_* 任务每窗口最少事件数 |
| `--min-actions` | 3 | action_* 任务每事件最少 action 数 |
| `--total-val` | 200 | 验证集总样本数（按 task 均分） |
| `--train-per-task` | -1 | 每任务最多训练样本数（-1 = 不限） |
| `--complete-only` | False | 跳过 clip 文件不存在的记录 |
| `--seed` | 42 | 随机数种子 |

---

## 答案重平衡

在 offline filter 之后，使用 `rebalance_aot_answers.py` 对保留的样本做选项重排，不丢样本：

```bash
python proxy_data/youcook2_seg/temporal_aot/rebalance_aot_answers.py \
  --input-jsonl $OUTPUT_DIR/train.offline_filtered.jsonl \
  --output-jsonl $OUTPUT_DIR/train.offline_filtered.balanced.jsonl
```

---

## 旧版管线（`legacy/`，已废弃）

> ⚠️ 以下脚本已移至 `legacy/` 子目录。它们依赖独立的 event clip 数据库和 VLM captioning 步骤，已由 `build_aot_from_seg.py` 完全替代，不再维护。

| 脚本 | 旧版职责 |
|------|---------|
| `build_event_aot_data.py` | 从 clip DB 构建 manifest + reverse/shuffle/composite 视频 |
| `annotate_event_captions.py` | VLM caption 生成（forward/reverse/shuffle） |
| `check_and_refine_captions.py` | caption 质量审核与精修 |
| `build_aot_mcq.py` | 基于 caption pairs 构建 MCQ（aot_v2t / aot_t2v / aot_4way_v2t） |
| `mix_aot_with_youcook2.py` | 与 YouCook2 时序分割数据混合 |
