# 三层分割 (Hier Seg) 消融实验

基于分层时序标注（L1 宏观阶段 / L2 事件检测 / L3 原子动作定位），通过 RL 训练消融验证最优层级组合。

---

## V1 训练数据构建 & Baseline 评估 (推荐入口)

### 整体流程

```
原始标注 JSON
    ↓ build_hier_data.py (per-phase L2, 筛选, 均衡采样)
JSONL (L1/L2/L3_seg 分别输出)
    ↓ prepare_clips.py (L1@1fps, L2@2fps, L3@2fps)
clipped JSONL + 物理视频 clips
    ↓ eval_baseline_rollout.py (采样 + rollout + reward 对比)
评估结果 (segment vs. segment+hint)
```

### 快速开始 — 一键构建 V1 数据

```bash
cd /path/to/train

# 默认: per-phase L2, 筛选, 领域均衡 800 条/层, 300 val
bash proxy_data/youcook2_seg/hier_seg_annotation/build_v1_data.sh

# 带 hint 版本 (prompt 附加 criterion 改写后的结构化 hint)
bash proxy_data/youcook2_seg/hier_seg_annotation/build_v1_data.sh --use-hint

# 自定义参数
BALANCE_PER_LEVEL=600 \
TRAIN_PER_LEVEL=600 \
L2_MIN_EVENTS=3 \
L3_MIN_ACTIONS=3 \
bash proxy_data/youcook2_seg/hier_seg_annotation/build_v1_data.sh
```

**环境变量 (可覆盖)**:

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VERSION` | `v1` | 输出子目录名 |
| `ANNOTATION_DIR` | common.sh 中的默认路径 | 标注 JSON 目录 |
| `BALANCE_PER_LEVEL` | `800` | 领域均衡采样目标 (每层) |
| `TRAIN_PER_LEVEL` | `800` | 每层 train 上限 |
| `TOTAL_VAL` | `300` | 总 val 数 (三层平分) |
| `L2_MODE` | `phase` | L2 模式: `phase` (per-phase) 或 `window` (滑窗) |
| `L1_MIN_PHASES` | `2` | L1 最少 phase 数 |
| `L2_MIN_EVENTS` | `2` | L2 最少 event 数 |
| `L3_MIN_ACTIONS` | `3` | L3 最少 action 数 |
| `USE_HINT` | `false` | 设为 `true` 等价于传 `--use-hint` |

**输出结构**:

```
${ABLATION_DATA_ROOT}/v1/
├── L1/
│   ├── train.jsonl              # 原始 JSONL (视频路径指向源文件)
│   ├── val.jsonl
│   ├── train_clipped.jsonl      # clip 后 JSONL (视频路径指向 clips/)
│   └── val_clipped.jsonl
├── L2/   (同上)
├── L3_seg/  (同上)
├── clips/
│   ├── L1/   {clip_key}_L1_1fps.mp4
│   ├── L2/   {clip_key}_L2_ph{id}_{start}_{end}.mp4
│   └── L3/   {clip_key}_L3_ev{id}_{start}_{end}.mp4
├── train_all.jsonl              # 三层合并 (train)
└── val_all.jsonl                # 三层合并 (val)
```

### build_hier_data.py 核心参数

```bash
python build_hier_data.py \
    --annotation-dir /path/to/annotations \
    --output-dir /path/to/output \
    --levels L1 L2 L3_seg \
    --l2-mode phase           # per-phase (推荐) 或 window (滑窗)
    --min-events 2            # L2 最少事件数
    --min-actions 3           # L3 最少动作数
    --balance-per-level 800   # 领域均衡采样 (-1=不做)
    --train-per-level 800     # 每层 train 上限
    --total-val 300           # 总 val 数
    --use-hint                # 使用 hint prompt (可选)
    --complete-only           # 仅处理三层完整的标注
```

**L2 两种模式**:

| 模式 | CLI | 输入 | 说明 |
|------|-----|------|------|
| **Per-phase** (推荐) | `--l2-mode phase` | 每个 L1 phase 作为独立输入 | 输出该 phase 内的 events，时间戳归零 |
| **Sliding window** | `--l2-mode window` | 128s 滑窗 (可配置) | 旧模式，兼容 Track 1 实验 |

**均衡采样逻辑** (`--balance-per-level N`):

1. 每层级独立处理
2. 按 `domain_l2` 分组，均匀分配 quota = N / n_domains
3. 不足 quota 的小域全部保留，多余名额重分配给大域
4. 大域内按 `output_count` 降序排列，优先保留 segment 数多的记录

### prepare_clips.py 核心参数

```bash
python prepare_clips.py \
    --input train.jsonl \
    --output train_clipped.jsonl \
    --clip-dir /path/to/clips \
    --l1-fps 1              # L1 帧率
    --l2l3-fps 2            # L2/L3 帧率 (0=stream copy)
    --workers 8             # 并行 ffmpeg 数
    --overwrite             # 覆盖已有输出
```

- L1: ffmpeg `-vf fps=1`，全视频重采样
- L2 (phase 模式): ffmpeg `-ss -t -vf fps=2`，按 phase 边界裁剪+重采样
- L3: ffmpeg `-ss -t` + 可选 `-vf fps=2`，按 event+padding 裁剪

### Hint 机制

标注 JSON 中有 3 对 criterion/hint 字段:

| 层级 | Criterion 字段 | Hint 字段 | JSON 位置 |
|------|---------------|-----------|-----------|
| L1 | `global_phase_criterion` | `global_phase_hint` | 顶层 |
| L2 | `event_split_criterion` | `event_split_hint` | 每个 phase 内 |
| L3 | `micro_split_criterion` | `micro_split_hint` | level3 顶层 |

- **criterion**: VLM 原始输出，包含具体视频内容描述
- **hint**: 经 LLM 改写后的内容无关版本，仅保留结构/逻辑信息
- `--use-hint` 时自动从 annotation JSON 读取 hint，附加到 prompt 末尾

### Baseline Rollout 评估

```bash
python eval_baseline_rollout.py \
    --input-dir /path/to/v1/ \
    --model-path /home/xuboshen/models/Qwen3-VL-4B-Instruct \
    --sample-per-level 50 \
    --num-rollouts 8 \
    --tensor-parallel-size 2 \
    --output-dir ./eval_results/

# 对比 hint 版本 (需要先用 --use-hint 构建第二份数据)
python eval_baseline_rollout.py \
    --input-dir /path/to/v1/ \
    --hint-input-dir /path/to/v1_hint/ \
    --model-path /home/xuboshen/models/Qwen3-VL-4B-Instruct \
    --sample-per-level 50 \
    --num-rollouts 8 \
    --output-dir ./eval_results/
```

**输出**:

```
eval_results/
├── segment_results.jsonl         # 逐条 rollout reward 详情
├── segment_hint_results.jsonl    # hint 版本详情 (如果有)
└── summary.json                  # 汇总统计 (per-level mean±std)
```

---

## 研究问题

**哪些层级的组合最能通过 RL 训练提升时序分割能力？** L3 的 query 顺序（正序/乱序）对训练有何影响？

## 三层任务设计

| 层级 | problem_type | 粒度 | 输入 | 输出 | Reward | 样本数 |
|------|-------------|------|------|------|--------|--------|
| L1 | `temporal_seg_hier_L1` | 阶段级 | 原始视频 (真实时间戳) | 秒数区间 3-5 段 | F1-IoU (NMS+Hungarian) | 500 |
| L2 | `temporal_seg_hier_L2` | 事件级 | 128s 滑窗片段 | 秒数区间 3-6 段 | F1-IoU (NMS+Hungarian) | 1898 |
| L3 (grounding) | `temporal_seg_hier_L3` | 动作级 | event clip + query list | 按序定位 3-8 段 | Position-aligned mean tIoU | 3568 (seq+shuf各1784) |
| L3 (seg) | `temporal_seg_hier_L3_seg` | 动作级 | event clip (无 query) | 检测所有原子动作 | F1-IoU (NMS+Hungarian) | ~376 |

### L3 两种模式

- **Grounding** (`L3_seq/L3_shuf/L3_both`): 给定 action query 列表，按序输出每个 action 的时间段。Reward 使用 position-aligned tIoU（pred[i] 对 gt[i]）
- **Segmentation** (`L3_seg`): 不给 query 文本，让模型自己检测所有原子动作。Reward 使用 F1-IoU（同 L1/L2 的匈牙利匹配），三层全部统一为分割任务

---

## 实验矩阵（7 组）

### 单层基线

| Exp | 名称 | 数据 | 样本数 | 说明 |
|-----|------|------|--------|------|
| 1 | `hier_seg_exp1_L2_only` | L2 | 1898 | 滑窗事件检测单任务基线 |
| 2 | `hier_seg_exp2_L3_seq` | L3 sequential | 1784 | 原子动作定位（正序）基线 |

### L3 顺序消融

| Exp | 名称 | 数据 | 样本数 | 说明 |
|-----|------|------|--------|------|
| 3 | `hier_seg_exp3_L3_shuf` | L3 shuffled | 1784 | 乱序是否增强/削弱定位 |
| 4 | `hier_seg_exp4_L3_both` | L3 seq+shuf | 3568 | 两种顺序混合是否互补 |

### 多层组合

| Exp | 名称 | 数据 | 样本数 | 说明 |
|-----|------|------|--------|------|
| 5 | `hier_seg_exp5_L2_L3` | L2 + L3(seq) | 3682 | 粗+细联合训练 |
| 6 | `hier_seg_exp6_L1_L2_L3` | L1+L2+L3(seq) | 4182 | 三层全联合 |
| 7 | `hier_seg_exp7_all_mixed` | L1+L2+L3(both) | 5966 | 最大数据量+顺序多样性 |

### 关键对比

| 对比 | 测试什么 |
|------|---------|
| exp2 vs exp3 | L3 正序 vs 乱序（顺序先验对训练的影响） |
| exp2 vs exp4 | L3 单顺序 vs 双顺序（数据多样性获益） |
| exp1 vs exp5 | L2 单独 vs L2+L3（细粒度任务是否帮助事件检测） |
| exp2 vs exp5 | L3 单独 vs L2+L3（上层任务是否帮助动作定位） |
| exp5 vs exp6 | 两层 vs 三层（L1 宏观分割的边际价值） |
| exp6 vs exp7 | 三层(seq) vs 三层(both)（乱序对全局训练的影响） |

---

## Reward 函数

使用 `youcook2_hier_seg_reward.py:compute_score`（**不是** `mixed_proxy_reward.py`），按 `problem_type` 分发：

| Level | Reward | 说明 |
|-------|--------|------|
| L1/L2 | F1-IoU | NMS 去重 + 匈牙利匹配 + F1 score |
| L3 | Position-aligned mean tIoU | pred[i] 对 gt[i]，分母取 max(n_pred, n_gt) |

---

## 共用超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Model | Qwen3-VL-4B-Instruct | |
| Algorithm | `ema_grpo` | 与 AOT/TG 对齐 |
| LR | 5e-7 | cosine warmup/decay |
| Online filtering | 启用 | |
| MAX_RESPONSE_LEN | 512 | multi-segment 输出 |
| Rollout N | 8 | |
| Max steps | 60 | |
| Val freq | 10 steps | |

完整参数见 [common.sh](common.sh)。

---

## 运行

```bash
cd /path/to/train

# 单层基线
bash local_scripts/hier_seg_ablations/exp1_L2_only.sh
bash local_scripts/hier_seg_ablations/exp2_L3_seq.sh

# L3 顺序消融
bash local_scripts/hier_seg_ablations/exp3_L3_shuf.sh
bash local_scripts/hier_seg_ablations/exp4_L3_both.sh

# 多层组合
bash local_scripts/hier_seg_ablations/exp5_L2_L3.sh
bash local_scripts/hier_seg_ablations/exp6_L1_L2_L3.sh
bash local_scripts/hier_seg_ablations/exp7_all_mixed.sh
```

数据自动准备：首次运行时 `prepare_data.py` 自动从 per-level JSONL 中筛选/split/merge。

> **新数据流 (推荐)**: `build_hier_data.py` 可直接从 annotation JSON 一步构建所有层级数据，
> 取代旧的 5 步流水线。Prompt 消融实验 (Track 2 V2 系统) 已迁移为先用 `build_hier_data.py`
> 构建基础数据，再用 `prepare_prompt_data.py` 做 prompt variant 替换。

---

## L3 数据迭代方案

### 当前状态
L3 标注基于 VLM 自动标注（Gemini），原子动作时间边界可能不精确。

### 迭代路线

1. **分割 → Grounding 转换**: 已实现 L3 seg 模式（`temporal_seg_hier_L3_seg`），三层统一为纯分割任务，均使用 F1-IoU reward
2. **Single Grounding**: 可进一步将 multi-grounding 拆成 single-query grounding，与 TG 任务 format 对齐，使用 iou_v2 reward
2. **Self-Training 校准**: 用训好的模型 rollout → 高 reward 样本替换原标注 → 迭代
3. **Hard Negative Mining**: 分析低 reward 高 variance 样本，人工校验/重标
4. **跨层一致性校验**: L3 时间段 ⊂ L2 event ⊂ L1 phase，检测修正不一致标注

---

## 文件结构

```
local_scripts/hier_seg_ablations/
├── README.md                          # 本文件
├── common.sh                          # 共用超参数 (含 ANNOTATION_DIR/CLIP_DIR_L2/L3)
├── build_hier_data.py                 # 统一数据构建: annotation JSON → JSONL (一步)
├── eval_baseline_rollout.py           # Baseline rollout 评估 (segment vs. hint)
│
│ ── 层级组合消融 (Exp 1-7) ──
├── prepare_data.py                    # 按层/变体筛选 + split + merge (从 _clipped.jsonl 读)
├── launch_train.sh                    # 统一训练入口
├── exp1_L2_only.sh ... exp7_all_mixed.sh
│
│ ── Prompt 消融 V2 (V1-V4) ──   ← 当前活跃版本
├── prompt_ablation/
│   ├── exp_v2_ablation.sh             # 单变体训练入口 (VARIANT=V1..V4, LEVELS=L1 L2 L3)
│   ├── run_v2_ablation.sh             # 批量运行 V1-V4
│   ├── prepare_prompt_data.py         # Prompt variant 替换 (从 build_hier_data.py 输出读)
│   └── prompt_variants_v2.py          # V1-V4 prompt 模板 (domain-agnostic, L1 时间戳模式)
│
│ ── Chain-of-Segment (V2 = ground-seg) ──
├── chain_seg_ablation/                # Chain-Seg 消融 (仅保留 V2: ground-seg)
│
│ ── 批量运行 ──
└── run_batch.sh

proxy_data/youcook2_seg/hier_seg_annotation/
├── build_v1_data.sh                   # [NEW] V1 一键构建脚本 (build + clip + merge)
├── prompts.py                         # prompt 模板库 (含 hint 版本)
├── prepare_clips.py                   # 物理视频截取 (含 phase-level clips)
├── visualize_annotations.py           # 数据分布可视化 (筛选+均衡采样+9 张图)
├── annotate.py / annotate_check.py    # 标注工具链
├── rewrite_criteria_hints.py          # criterion → hint 改写
├── build_dataset.py                   # [DEPRECATED] 已被 build_hier_data.py 替代
├── sample_mixed_dataset.py            # [DEPRECATED]
└── run_build.sh                       # [DEPRECATED]

proxy_data/shared/
└── seg_source.py                      # 统一常量/加载/clip 命名 (含 get_l2_phase_clip_path)

verl/reward_function/
├── youcook2_hier_seg_reward.py        # L1/L2/L3_seg: F1-IoU
└── youcook2_chain_seg_reward.py       # Chain-of-Segment V2: 0.4*tIoU + 0.6*F1-IoU
```

---

## 消融实验 Track 2: Prompt 消融 V2 (2×2 Factorial)

### 研究问题

**Prompt 的详细程度（粒度定义）和 CoT 推理对多层分割质量有何影响？**

### 四种 Prompt 变体 (V1-V4)

|  | No CoT | CoT (`<think>`) |
|---|---|---|
| **Minimal** | V1 (baseline) | V3 |
| **Granularity-Enhanced** | V2 | V4 |

| Variant | 描述 | 关键差异 |
|---------|------|----------|
| **V1** | 当前基线 — 简短任务描述 + 格式示例 | 无粒度定义，无 CoT |
| **V2** | 加入粒度定义 — 事件时长范围、反碎片化规则、段数范围 | 详细定义但无推理 |
| **V3** | 加入 CoT — 要求先 `<think>` 描述观察再输出 `<events>` | 简短定义但有推理 |
| **V4** | 粒度定义 + CoT（V2 + V3） | 详细定义 + 推理 |

### 关键对比

| 对比 | 测试什么 |
|------|---------|
| V1 vs V2 | 粒度定义是否减少碎片化/过粗分割 |
| V1 vs V3 | CoT 推理是否提升定位精度 |
| V2 vs V4 | 在粒度定义基础上加 CoT 是否有增益 |
| V3 vs V4 | 在 CoT 基础上加粒度定义是否有增益 |
| V1 vs V4 | 两个优化同时加是否最优 |

### 运行

```bash
# 单个变体（三层均衡, 400 train + 100 val per level）
VARIANT=V2 LEVELS="L1 L2 L3" bash local_scripts/hier_seg_ablations/prompt_ablation/exp_v2_ablation.sh

# 全部四个变体
LEVELS="L1 L2 L3" bash local_scripts/hier_seg_ablations/prompt_ablation/run_v2_ablation.sh

# 仅 L2
VARIANT=V2 LEVELS="L2" bash local_scripts/hier_seg_ablations/prompt_ablation/exp_v2_ablation.sh
```

### 注意

- V3/V4 (CoT) 变体会自动将 `MAX_RESPONSE_LEN` 设为 1024（`<think>` 需要额外 token）
- GT answer 仍为纯 `<events>` 格式（不含 `<think>`），reward 只看 `<events>` 标签
- 数据流: `build_hier_data.py` 先构建基础数据 → `prepare_prompt_data.py` 替换 prompt variant
- L3 使用自由分割模式（`L3_seg`），三层统一用 F1-IoU reward

---

## 消融实验 Track 3: Chain-of-Segment (链式层次分割)

### 研究问题

**让模型"先粗后精"（先 grounding L2 事件，再 segment L3 原子动作）是否优于独立训练 L2 和 L3？**

### 任务设计

```
输入: 128s 视频片段 + L2 事件描述列表（caption 文本）
      ↓
Step 1: L2 Grounding — 定位每个事件的 [start, end]
      ↓
Step 2: L3 Segmentation — 每个事件内识别原子动作 [[s1,e1], [s2,e2], ...]
      ↓
输出: <l2_events>[[...]]</l2_events>
      <l3_events>[[[...]], [[...]]]</l3_events>
```

### Reward 公式

$$R = 0.4 \cdot tIoU_{L2} + 0.6 \cdot F1\text{-}IoU_{L3}$$

| 分项 | 计算方式 | 说明 |
|------|---------|------|
| $tIoU_{L2}$ | Position-aligned mean tIoU | caption 给出顺序，pred[i] 对 gt[i] |
| $F1\text{-}IoU_{L3}$ | Per-event mean F1-IoU | 每个 L2 事件内的 L3 段用 F1-IoU (NMS+Hungarian) |

### 数据来源

通过 `chain_seg_ablation/build_chain_seg_data.py` 直接从原始标注 JSON 构建：
- 直接读取 annotations/*.json，通过 `parent_event_id` 关联 L2↔L3
- L2 caption 直接取自 `level2.events[i].instruction`
- ≥2 个 L2 事件有 L3 对应 (min_actions ≥ 3) → 组成一条样本
- L3 时间戳自动转换为 L2 窗口相对坐标

| 统计 | 值 |
|------|-----|
| 总样本 | 737 (590 train + 147 val) |
| 平均事件数/样本 | 2.5 |

### 关键对比

| 对比 | 测试什么 |
|------|---------|
| exp8 vs exp5 | 链式推理 vs 独立 L2+L3 混合训练 |
| exp8 vs exp1 | 多级输出 vs 纯 L2 事件检测 |

### 运行

```bash
bash local_scripts/hier_seg_ablations/exp8_chain_L2L3.sh
```

---

## 三条消融 Track 总览

| Track | 研究问题 | 实验数 | 变量 |
|-------|---------|--------|------|
| **Track 1**: 层级组合 | 哪些层级组合最优？L3 顺序的影响？ | 7 (exp1-7) | 层级 × L3顺序 |
| **Track 2**: Prompt 消融 V2 | 粒度定义 & CoT 对多层分割的影响？ | 4 (V1-V4) | 粒度 × CoT |
| **Track 3**: Chain-of-Segment | "先粗后精" 链式推理 vs 独立训练？ | 1 (exp8) | 任务结构 |

### 数据流程图

```
原始标注 JSON (annotations/*.json)
    ↓
    ├─── build_v1_data.sh (proxy_data/.../hier_seg_annotation/) ──→ V1 基础训练数据
    │    ├ build_hier_data.py (JSONL, per-phase L2, 均衡采样)
    │    └ prepare_clips.py (L1@1fps, L2@2fps, L3@2fps)
    │
    ├─── eval_baseline_rollout.py ──────────→ Baseline 评估 (segment vs hint)
    │
    ├── build_hier_data.py + prepare_prompt_data.py → Track 2: Prompt 消融数据 (V1-V4)
    ├── prepare_data.py (从 _clipped.jsonl 读)          → Track 1: 层级组合数据
    └── chain_seg_ablation/build_chain_seg_data.py       → Track 3: Chain-Seg 数据
    ↓
EasyR1 训练 (launch_train.sh)
    ↓ (reward dispatch)
    ├── youcook2_hier_seg_reward.py   → Track 1 & 2 + V1 的 reward (F1-IoU)
    └── youcook2_chain_seg_reward.py  → Track 3 的 reward (0.4*tIoU + 0.6*F1-IoU)
    ↓
WandB / TensorBoard 对比
```
