# 两阶段 LLM 数据筛选 Pipeline 汇报

> 文档目的：在启动全量 Stage A 前，供 check 筛选流程、提示词设计、决策逻辑是否合理。

---

## 1. 为什么要两阶段？

旧方案（`assess_hierarchy.py`）在单次 LLM 调用中同时评估粒度、L1/L3 潜力、时序结构、综合分数，导致 **补偿效应**：某个维度高分会拉高整体评分，即便粒度不匹配。

新方案将评估拆为两步：

```
text_filter.py (纯文本规则) → Stage A (粒度判断) → Stage B (层次潜力评估)
         ↓                          ↓                        ↓
      ~10K passed               keep/maybe/reject         keep/maybe/reject
```

- **Stage A 只管粒度**：这些标注是 L2-like 吗？
- **Stage B 只管潜力**：能否支撑 L1 聚合 + L3 分解？

两阶段解耦后，不会因为 "L1 潜力高" 而放过粒度不匹配的样本。

---

## 2. 数据流全景

```
ET-Instruct-164K (164K)   TimeLens-100K (19K)
         │                         │
    text_filter.py            text_filter.py
    (60-240s, ≥5 events)      (60-240s, ≥5 events)
         │                         │
     ~10K passed              ~12K passed
         │                         │
  ┌──────┴──────┐          ┌──────┴──────┐
  │  Stage A    │          │  Stage A    │
  │ 粒度粗筛    │          │ 粒度粗筛    │
  └──────┬──────┘          └──────┬──────┘
   keep │ maybe │ reject    keep │ maybe │ reject
         │                         │
  ┌──────┴──────┐          ┌──────┴──────┐
  │  Stage B    │          │  Stage B    │
  │ 层次潜力    │          │ 层次潜力    │
  └──────┬──────┘          └──────┬──────┘
   keep │ maybe │ reject    keep │ maybe │ reject
         │                         │
    最终候选池              最终候选池
```

---

## 3. Stage A：L2 粒度粗筛

### 3.1 设计目标

判断已有标注的粒度是否适合作为 L2 事件，**不假设标注天然就是 L2**。

### 3.2 System Prompt

```
You are evaluating whether provided temporal annotations from a video dataset
can serve as Level-2 (L2) event annotations in a 3-level hierarchical temporal
segmentation framework.

Hierarchy:
- L1: broad macro phases, each covering multiple related sub-goals
- L2: goal-directed local task units with meaningful sub-goals
- L3: short atomic actions or visible state-change steps

Important:
Do NOT assume the provided annotations are valid L2 events.
Judge whether their granularity is mostly L1-like, mostly L2-like,
mostly L3-like, or mixed.
Be conservative: if uncertain whether they are truly L2-like, prefer
"mixed" or "reject".

A good L2 segment:
- corresponds to one meaningful local sub-goal
- is larger than a single short action
- is smaller than a broad multi-subgoal stage

Use these criteria for L2 fit:
- If many segment descriptions are short action phrases like "pick up",
"pour", "cut", "place", or similar near-atomic steps, the sample is
likely too fine.
- If many segment descriptions summarize broad processes like "prepare
ingredients", "cook the dish", "make the sauce", or similar multi-subgoal
stages, the sample is likely too coarse.
- If uncertain, err on the side of "mixed" or "reject" rather than "keep".

## Examples

### Example 1 — mostly_L2_like (keep)

Input:
Video duration: 180.0s | Domain: coin | Segments: 7
  1. [0.0s - 23.5s] Spread glue evenly on the wood surface
  2. [23.5s - 48.0s] Attach decorative veneer to the glued surface
  ...

Output:
{"reasoning":"Each segment covers one distinct sub-goal in a woodworking
process with 20-30s durations — clearly goal-directed L2 units, not atomic
actions nor broad phases.","granularity_label":"mostly_L2_like",
"l2_fit_score":5,"granularity_issue":"good","mixed_ratio_estimate":"low",
"decision":"keep"}

### Example 2 — mostly_L3_like (reject)

Input:
Video duration: 120.0s | Domain: activitynet | Segments: 9
  1. [0.0s - 5.2s] Pick up the eggs from the counter
  2. [5.2s - 8.0s] Crack each egg into a bowl
  ...

Output:
{"reasoning":"Segments are very short (3-7s) single atomic actions (pick up,
crack, add, whisk). These are L3 fine-grained steps, not L2 goal-directed
units.","granularity_label":"mostly_L3_like","l2_fit_score":1,
"granularity_issue":"too_fine","mixed_ratio_estimate":"low","decision":"reject"}

Return only valid JSON.
```

### 3.3 User Prompt（填入具体样本数据）

```
Video duration: {duration}s
Domain: {source}
Number of annotated segments: {n_events}

Annotated segments:
  1. [12.0s - 36.0s] clean the bananas and cut them into pieces
  2. [36.0s - 44.0s] heat oil in a pan
  ...

Evaluate whether these provided segments are at the right granularity to
serve as L2 events.

Respond with ONLY valid JSON:
{
  "reasoning": "<1-2 short sentences>",
  "granularity_label": "mostly_L1_like | mostly_L2_like | mostly_L3_like | mixed",
  "l2_fit_score": <1-5>,
  "granularity_issue": "too_coarse | good | too_fine | mixed",
  "mixed_ratio_estimate": "low | medium | high",
  "decision": "keep | maybe | reject"
}
```

### 3.4 输出字段说明

| 字段 | 类型 | 含义 |
|------|------|------|
| `granularity_label` | enum | 粒度标签: mostly_L1_like / mostly_L2_like / mostly_L3_like / mixed |
| `l2_fit_score` | 1-5 | L2 适配度打分 |
| `granularity_issue` | enum | 粒度问题: too_coarse / good / too_fine / mixed |
| `mixed_ratio_estimate` | enum | 混合粒度比例: low / medium / high |
| `decision` | enum | keep / maybe / reject |
| `reasoning` | str | 1-2 句理由 |

### 3.5 程序化决策规则（实时覆盖 LLM）

规则在 `assess_sample()` 中 **实时** 应用：LLM 返回后立即调用 `apply_stage_a_rules()`，若与 LLM decision 不一致则覆写（原始决定存入 `_original_decision`）。

```python
# 硬 reject:
if label in ("mostly_L1_like", "mostly_L3_like"):  → reject
if l2_fit_score <= 2:                               → reject

# 硬 keep:
if (label == "mostly_L2_like"
    and score >= 4
    and issue == "good"
    and mixed_ratio == "low"):                      → keep

# 灰区:
if label == "mixed":                                → maybe
if score == 3:                                      → maybe
if label == "mostly_L2_like" and score >= 4:        → keep (soft)
```

---

## 4. Stage B：层次潜力精筛

### 4.1 设计目标

对 Stage A keep 的样本评估是否能支撑完整的三层标注：
- **L1 聚合潜力**：事件能否分组为 2-4 个宏观阶段？
- **L3 分解潜力**：事件描述是否暗示可分解的子动作？
- **时序结构**：事件间是否有清晰的时序逻辑？

### 4.2 System Prompt

```
You are evaluating whether a set of temporal annotations that are already
roughly L2-like can support a full 3-level hierarchical segmentation setup.

Hierarchy:
- L1: macro phases grouping multiple events
- L2: provided event-like segments (candidate L2 annotations)
- L3: atomic actions inside each event

Assume the provided segments are candidate L2 annotations, but still judge
strictly. Focus on whether they support meaningful L1 grouping AND L3
decomposition.

Evaluation criteria:
- L1 Potential: Can these events be naturally grouped into 2-4 macro phases
with distinct themes? (5=clear phases, 1=events are all independent)
- L3 Potential: Do event descriptions suggest decomposable sub-actions?
(5=rich detail implying multiple steps, 1=events are already near-atomic)
- Temporal Structure: Are events well-ordered with clear temporal flow?
(5=strong narrative arc, 1=random/overlapping/unclear progression)
- Overall: Combined suitability for a production-quality 3-level annotation

Be strict: only give overall_score >= 4 if ALL three dimensions score >= 3.

## Examples

### Example 1 — strong hierarchical potential (keep)

Input:
Video duration: 200.0s | Domain: coin | Segments: 6
  1. [0.0s - 28.0s] Measure and mark the wood pieces for cutting
  2. [28.0s - 58.0s] Cut the wood along marked lines with a saw
  ...

Output:
{"reasoning":"Clear 3 macro phases emerge (Preparation: 1-2, Assembly: 3-5,
Finishing: 6). Each event implies multiple sub-actions. Strong temporal
flow.","l1_potential":5,"l3_potential":4,"temporal_structure":5,
"overall_score":5,"phase_sketch":["Preparation: 1,2","Assembly: 3,4,5",
"Finishing: 6"],"decision":"keep"}

### Example 2 — poor hierarchical potential (reject)

Input:
Video duration: 150.0s | Domain: queryd | Segments: 5
  1. [10.0s - 35.0s] A person talks about their morning routine
  2. [40.0s - 65.0s] The same person describes their favorite food
  ...

Output:
{"reasoning":"Talking-head video with no physical actions. L3 decomposition
impossible — each segment is just continuous speech. No meaningful temporal
structure.","l1_potential":2,"l3_potential":1,"temporal_structure":2,
"overall_score":1,"phase_sketch":["Monologue: 1,2,3,4,5"],"decision":"reject"}

Return only valid JSON.
```

### 4.3 User Prompt

```
Video duration: {duration}s
Domain: {source}
Number of candidate L2 segments: {n_events}

Candidate L2 segments:
  1. [12.0s - 36.0s] clean the bananas and cut them into pieces
  2. [36.0s - 44.0s] heat oil in a pan
  ...

Evaluate this sample for hierarchical segmentation suitability.

Respond with ONLY valid JSON:
{
  "reasoning": "<1-2 short sentences>",
  "l1_potential": <1-5>,
  "l3_potential": <1-5>,
  "temporal_structure": <1-5>,
  "overall_score": <1-5>,
  "phase_sketch": ["phase_name: event_indices", ...],
  "decision": "keep | maybe | reject"
}
```

### 4.4 输出字段说明

| 字段 | 类型 | 含义 |
|------|------|------|
| `l1_potential` | 1-5 | L1 聚合潜力（事件能否分组为宏观阶段） |
| `l3_potential` | 1-5 | L3 分解潜力（事件是否可分解子动作） |
| `temporal_structure` | 1-5 | 时序结构（事件间先后逻辑是否清晰） |
| `overall_score` | 1-5 | 综合评分 |
| `phase_sketch` | list[str] | L1 阶段草图，如 `["Preparation: 1-3", "Cooking: 4-7"]` |
| `decision` | enum | keep / maybe / reject |
| `reasoning` | str | 1-2 句理由 |

### 4.5 程序化决策规则（实时覆盖 LLM）

与 Stage A 相同，规则在 `assess_sample()` 中实时应用（`apply_stage_b_rules()`），不一致时覆写并记录 `_original_decision`。

```python
# 硬 reject:
if overall_score <= 2:                              → reject
if any(dim <= 1 for dim in [l1, l3, temporal]):     → reject

# 硬 keep:
if overall_score >= 4 and all(dim >= 3):            → keep

# 灰区:
if overall_score == 3:                              → maybe
if overall_score >= 4 but some dim < 3:             → maybe
```

---

## 5. 关于你的问题：偏 L3 的标注怎么处理？

**核心问题**：有些样本的标注虽然偏 L3（near-atomic），但视频本身内容很好，值得保留。

**当前设计的处理方式**：

| Stage A 判断 | 处理 |
|--|--|
| `mostly_L3_like` | **硬 reject** — 粒度太细，无法作为 L2 |
| `mixed` + L3 比例 high | **maybe** — 灰区，需人工看 |
| `mixed` + L3 比例 medium/low | **maybe** — 有机会 |
| `mostly_L2_like` | **keep** — 粒度合适 |

**可能的改进方案（供讨论）**：

1. **方案 A：保留 mixed 中的好样本** — 当前 `maybe` 样本不进 Stage B，但可以进一步处理：
   - 对 `maybe` 中 `l2_fit_score=3` 且 `mixed_ratio=medium` 的样本也进 Stage B
   - 让 Stage B 决定最终去留
   - **这样做的代价**：Stage B 处理量增加（可能从 keep 的 ~30-50% 扩大到 ~60-70%）

2. **方案 B：L3-like 标注的 "升级" 路线** — 对判定为 L3-like 的好视频，不 reject 而是标记 `_needs_regrouping=True`：
   - 这些样本的原始标注改为 L3 候选
   - 后续流程中需要额外生成 L2 标注（把多个 L3 合并为 L2）
   - **好处**：不浪费好视频
   - **代价**：需要额外的 "L3→L2 合并" 标注流程

3. **方案 C：放宽 mixed 阈值** — 把 mixed + l2_fit_score=3 从 `maybe` 改为 `keep`（带风险标注）

**建议**：先全量跑完 Stage A 看 maybe 的分布。如果 maybe 占比大（>30%），可以考虑方案 A 让 Stage B 也处理 maybe。

---

## 6. 开销估算

### ET-Instruct-164K

| 阶段 | 样本数 | 输入 tokens | 输出 tokens | 预估费用 | 时间(16w) |
|------|--------|-------------|-------------|----------|-----------|
| Stage A | ~10K | ~4.5M | ~1.2M | **~$17** | ~10min |
| Stage B (est. 40% keep) | ~4K | ~1.8M | ~0.5M | **~$8** | ~4min |
| **合计** | - | ~6.3M | ~1.7M | **~$25** | ~14min |

### TimeLens-100K

| 阶段 | 样本数 | 输入 tokens | 输出 tokens | 预估费用 | 时间(16w) |
|------|--------|-------------|-------------|----------|-----------|
| Stage A | ~12K | ~5.4M | ~1.4M | **~$20** | ~12min |
| Stage B (est. 40% keep) | ~5K | ~2.3M | ~0.6M | **~$10** | ~5min |
| **合计** | - | ~7.7M | ~2.0M | **~$30** | ~17min |

### 总计

| | 样本数 | 费用 | 时间 |
|---|---|---|---|
| **Stage A 全量** | ~22K | ~$37 | ~22min |
| **Stage B** | ~9K | ~$18 | ~9min |
| **Pipeline 合计** | ~22K | ~$55 | ~31min |

*假设 Novita Gemini 2.5 Pro: ~$2/M input, ~$12/M output。实际可能偏差 ±30%。*

---

## 7. 当前设计的漏筛分析与后续规划

### 7.1 核心局限：标注粒度 ≠ 视频质量

当前 pipeline 的筛选逻辑是 **"标注粒度 → 视频去留"**，本质上是在判断 **现有标注是否恰好是 L2 粒度**，而非 **视频内容本身是否适合层次分割**。这导致一个系统性偏差：

| 视频内容质量 | 标注粒度 | 当前 pipeline 行为 | 是否应该保留？ |
|---|---|---|---|
| 好（多事件、状态变化丰富） | L2-like | keep | 是 |
| 好 | L3-like（标注过细） | **reject** | **是** — 漏筛 |
| 好 | L1-like（标注过粗） | **reject** | **是** — 漏筛 |
| 好 | 标注少（<5 events） | **text_filter reject** | **可能** — 原标注覆盖不全 |
| 差（单调、无结构） | 任意 | 可能 keep | 否 — 误保留 |

**漏筛主要来源**：

1. **L3-like 好视频被硬 reject** — Section 5 已讨论，最大漏筛源。例如 cooking 视频，原标注是 "pick up knife" / "cut onion" / "place in bowl" 这种 near-atomic 动作，但视频本身完全可以支撑 L2 分割。Stage A 会给 `mostly_L3_like → reject`。

2. **标注事件数不足但视频丰富** — text_filter 要求 ≥5 events，但很多数据集标注稀疏（如 tvg 任务通常只标注 1 个事件、vhd 只标注高光段）。视频可能有 10+ 个可分割事件，但因为标注只覆盖了 1-3 个，在 text_filter 阶段就被丢弃。

3. **Maybe 灰区无出路** — Stage A 的 `mixed` 样本进入 maybe 后不进 Stage B，直接沉没。如果 mixed 占比大（>30%），损失显著。

4. **标注格式决定粒度** — ET-Instruct 的标注粒度取决于原始 task 类型（slc 偏 L3、dvc 偏 L2、tal 变化大），与视频本身无关。同一个视频在不同 task 下可能被标注出完全不同粒度的事件。

### 7.2 粗筛定位：低成本漏斗，不求召回率

当前 pipeline 的正确定位是 **低成本文本粗筛**，目标是快速找到 "标注本身就接近 L2" 的幸运样本，而非追求高召回率。

预估漏筛规模（粗略估计）：

| 阶段 | 输入 | 通过率 | 漏筛的好视频（估） |
|---|---|---|---|
| text_filter | 164K + 19K | ~12% | 大量（标注稀疏的好视频被 min_events 过滤） |
| Stage A | ~22K | ~40% keep | 中等（L3-like / L1-like 好视频被 reject） |
| Stage B | ~9K | ~50% keep | 少量（到这步基本是精准筛选） |
| **最终候选** | ~4-5K | — | — |

从 183K 原始样本到 ~4-5K 候选，通过率约 2.5%。即便假设只有 10% 的原始视频真正适合层次分割（~18K），当前 pipeline 的召回率也仅约 25-30%。

### 7.3 后续计划：DVC 重标注弥补漏筛

仅靠文本粗筛无法覆盖那些 **视频内容好但标注不匹配** 的样本。计划引入 DVC（Dense Video Captioning）作为第二阶段：

```
当前 pipeline (文本粗筛)          DVC 补救路线
────────────────────          ──────────────
text_filter → Stage A → B     被 reject 的"好视频"子集
         │                           │
    ~4-5K 候选                  VLM DVC 重标注
         │                    （生成标准化 L2 事件）
         │                           │
         └───────── 合并 ──────────────┘
                    │
              最终候选池
```

DVC 路线的关键问题：
- **成本**：对大量视频跑 VLM DVC 比纯文本 LLM 筛选贵 10-100x
- **筛选哪些视频做 DVC**？需要一个视频级别的质量预判（duration、视觉多样性、场景变化等）
- **DVC 标注质量**：VLM 生成的事件是否真正是 L2 粒度？可能需要 DVC 后再过一遍 Stage A

### 7.4 关于 decision_rules 的集成说明

Section 3.5 和 4.5 描述的程序化决策规则（`decision_rules.py`）现已 **实时集成** 到 `assess_sample()` 中：

- `stage_a_coarse_filter.py` 和 `stage_b_fine_filter.py` 在每次 LLM 调用后立即调用 `apply_stage_a/b_rules()` 覆写 decision
- 若 LLM decision 与规则 decision 不一致，原始值存入 `_original_decision` 字段
- `print_stats()` 会汇报规则覆盖的总数和具体转换（如 `keep → maybe: 5`）
- `decision_rules.py` 仍可作为独立 CLI 工具用于离线分析

---

## 8. 文件清单

```
proxy_data/data_curation/
├── PIPELINE_REPORT.md              # 本文档
├── README.md                       # 数据源概述
├── configs/
│   ├── et_instruct_164k.yaml        # 筛选参数配置
│   └── timelens_100k.yaml
├── sources/
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── llm_client.py            # API 调用、并发、断点续评
│   │   ├── decision_rules.py        # 程序化决策规则
│   │   ├── stage_b_fine_filter.py   # Stage B (数据源无关)
│   │   ├── analyze_results.py       # 结果统计分析 + HTML 报告
│   │   └── convert_to_viz.py        # 转可视化格式 + 抽帧
│   ├── et_instruct_164k/
│   │   ├── text_filter.py           # Step 0: 文本规则筛选
│   │   ├── stage_a_coarse_filter.py # Stage A (ET-Instruct 专用)
│   │   ├── run_pipeline.sh          # 一键运行
│   │   └── results/                 # 输出目录
│   └── timelens_100k/
│       ├── text_filter.py
│       ├── stage_a_coarse_filter.py # Stage A (TimeLens 专用)
│       ├── run_pipeline.sh
│       └── results/
```

---

## 9. 运行命令（从 train/ 目录）

### ET-Instruct 全量 Stage A

```bash
python proxy_data/data_curation/sources/et_instruct_164k/stage_a_coarse_filter.py \
    --input proxy_data/data_curation/sources/et_instruct_164k/results/passed.jsonl \
    --output proxy_data/data_curation/sources/et_instruct_164k/results/stage_a_results.jsonl \
    --no-sample --workers 16
```

### 查看 Stage A 结果

```bash
python proxy_data/data_curation/sources/shared/analyze_results.py \
    --input proxy_data/data_curation/sources/et_instruct_164k/results/stage_a_results.jsonl \
    --stage A --review 3 \
    --html proxy_data/data_curation/sources/et_instruct_164k/results/stage_a_report.html
```

### Stage B（Stage A 完成后）

```bash
python proxy_data/data_curation/sources/shared/stage_b_fine_filter.py \
    --input proxy_data/data_curation/sources/et_instruct_164k/results/stage_a_results_keep.jsonl \
    --output proxy_data/data_curation/sources/et_instruct_164k/results/stage_b_results.jsonl \
    --data-source et_instruct --no-sample --resume --workers 16
```
