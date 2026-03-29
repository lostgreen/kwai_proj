# LLM + VLM 数据筛选 Pipeline 汇报

> 文档目的：供 check 筛选流程、提示词设计、决策逻辑是否合理。
> 版本: v4 — Source Routing + VLM Vision Filter

---

## 1. Pipeline 设计理念

### 1.1 为什么需要 Source Routing？

不同数据源的标注质量和粒度差异巨大：
- **Dense Manual**（coin, activitynet_captions 等）：有 5-20 个精细文本标注 → 可以判断边界清晰度和阶段多样性
- **Coarse Manual**（activitynet, hacs 等）：只有 1-3 个粗标签 → 只能判断活动本身是否物理丰富
- **ASR/Auto**（how_to_step, queryd 等）：时间戳不可靠 → 忽略时间戳，只看文本是否描述多步骤操作

用同一个 prompt 评估所有源会导致**评估维度不匹配**，特别是 Group B/C 的样本会因标注稀疏而被误杀。

### 1.2 为什么加 VLM 视觉校验？

纯文本 LLM 筛选有系统性盲区：
- 文本说"cooking"但视频可能是静态 talking head
- 复杂标注不代表视频本身内容丰富
- ASR 文本可能与实际画面不匹配

VLM 视觉校验作为**安全网**，在文本筛选 keep 之后，用 6 帧抽样检查视频是否真正包含物理活动和状态变化。

---

## 2. 数据流全景

```
ET-Instruct-164K (164K)              TimeLens-100K (19K)
         │                                    │
    text_filter.py                       text_filter.py
    (60-240s, ≥5 events)                 (60-240s, ≥5 events)
         │                                    │
     ~10K passed                         ~12K passed
         │                                    │
  ┌──────┴──────┐                     ┌──────┴──────┐
  │  Stage A    │                     │  Stage A    │
  │ Source      │                     │ Boundary +  │
  │ Routing     │                     │ Phase       │
  │ (3 Groups)  │                     │ Diversity   │
  └──────┬──────┘                     └──────┬──────┘
   keep │ reject                       keep │ reject
         │                                    │
  ┌──────┴──────┐                     ┌──────┴──────┐
  │ VLM Vision  │                     │ VLM Vision  │
  │ Filter      │                     │ Filter      │
  │ (6 frames)  │                     │ (6 frames)  │
  └──────┬──────┘                     └──────┬──────┘
   keep │ reject                       keep │ reject
         │                                    │
    最终候选池                           最终候选池
```

> **Stage B（层次潜力精筛）** 保留但当前不使用。可后续对 VLM keep 样本做更精细评估。

---

## 3. Stage A：文本粗筛

### 3.1 ET-Instruct — Source Routing（3 组策略）

基于 `source` 字段将样本路由到不同的评估策略：

| Group | Sources | 评估维度 | 通过阈值 |
|-------|---------|---------|---------|
| **A — Dense Manual** | coin, activitynet_captions, tacos, didemo, charades_sta | boundary_clarity + phase_diversity | diversity >= 3 |
| **B — Coarse Manual** | activitynet, hacs, thumos14 | physical_richness | richness >= 3 |
| **C — ASR/Auto** | how_to_step, how_to_caption, queryd, ego4d_naq, naq | action_density + is_physical_demo | density >= 4 |

未匹配的 source 默认走 Group A。

#### Group A Prompt（Dense Manual）

评估**边界清晰度 + 阶段多样性**。适合有 5+ 个精细标注的样本。

```json
{
  "structural_analysis": "<1-2 sentences>",
  "boundary_clarity_score": 1-5,
  "phase_diversity_score": 1-5,
  "decision": "keep | reject"
}
```

核心判断：视频是否有 2+ 个不同主题的「章节」（如 prep → cook → serve），而非同一动作的单调循环。

#### Group B Prompt（Coarse Manual）

评估**物理丰富度**。适合只有 1-3 个粗标签的样本（如 "playing basketball"）。

```json
{
  "physical_analysis": "<1-2 sentences>",
  "physical_richness_score": 1-5,
  "decision": "keep | reject"
}
```

核心判断：活动本身是否涉及多种工具/物体/动作（如 "washing car" → rinse, soap, scrub, dry），而非单调重复运动（如 "jogging"）。

#### Group C Prompt（ASR/Auto）

评估**文本是否描述多步骤物理操作**。**忽略时间戳**。

```json
{
  "text_analysis": "<1-2 sentences>",
  "action_density_score": 1-5,
  "is_physical_demo": true | false,
  "decision": "keep | reject"
}
```

核心判断：文本是否描述 hands-on 操作（cooking, crafting, repairing），而非 vlog/gaming/lecture/interview。阈值更严（>=4）因为 ASR 文本噪声大。

### 3.2 TimeLens — 统一评估

TimeLens 7 个 domain（cosmo_cap, internvid_vtime, didemo, queryd, hirest_step, hirest_grounding, hirest）的标注格式统一，使用**边界清晰度 + 阶段多样性**统一评估：

```json
{
  "structural_analysis": "<1-2 sentences>",
  "boundary_clarity_score": 1-5,
  "phase_diversity_score": 1-5,
  "decision": "keep | reject"
}
```

决策规则：`boundary_clarity >= 3 AND phase_diversity >= 3 → keep`，否则 `reject`。

### 3.3 程序化决策规则

规则在 `assess_sample()` 中**实时**应用：LLM 返回后立即检查，若与 LLM decision 不一致则覆写（原始决定存入 `_original_decision`）。

**ET-Instruct（per-group inline rules）**：
```python
Group A: phase_diversity_score >= 3          → keep
Group B: physical_richness_score >= 3        → keep
Group C: action_density_score >= 4           → keep
其余 / parse error                            → reject
```

**TimeLens**（`decision_rules.py:apply_richness_rules()`）：
```python
boundary_clarity >= 3 AND phase_diversity >= 3  → keep
其余 / parse error                                → reject
```

---

## 3.5 VLM 视觉校验（Safety Net）

对 Stage A keep 的样本，用 VLM 查看实际视频帧做**视觉安全网检查**。

### 3.5.1 帧抽取

`video_sampler.py` — 轻量级抽帧工具：
- 均匀抽取 6 帧
- 跳过开头 5% 和结尾 5%（避开片头片尾）
- 压缩至最长边 512px，JPEG base64 编码
- 依赖 decord + Pillow

### 3.5.2 VLM Prompt

发送 6 帧给 VLM，判断视频是否包含**真实世界物理活动**：

```json
{
  "visual_analysis": "<1-2 sentences>",
  "visual_quality_score": 1-5,
  "is_physical_activity": true | false,
  "decision": "keep | reject"
}
```

**Reject 场景**：
- Talking head / 新闻主播
- 静态场景（PPT、截屏、桌面）
- 游戏 / 动画 / 非真实物理场景
- 全帧画面几乎一样（无进展）

**决策规则**：`visual_quality_score >= 3 AND is_physical_activity == true → keep`

---

## 4. Stage B：层次潜力精筛（可选，当前不使用）

> **注意**：v2 设计中 Stage A 已改为视频内容潜力评估（单阶段筛选），通常不再需要 Stage B。
> Stage B 代码保留供需要时使用（如对 keep 样本做更细致的 L1/L3 分解能力评估）。

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

## 5. 设计演化记录

Pipeline 经历了 4 个主要版本：

| 版本 | 设计 | 问题 |
|------|------|------|
| v1 | L2 粒度分类 (granularity_label) | 96% reject — 问错了问题 |
| v2 | 视频内容丰富度评估 (3 维 scores) | >90% keep — 太宽松 |
| v3 | 边界清晰度 + 阶段多样性 (2 维) | 留存率合理但不区分数据源特点 |
| **v4 (当前)** | Source Routing + VLM Vision Filter | 按数据源特点评估 + 视觉安全网 |

**v4 核心改进**：
- 认识到不同数据源标注质量差异巨大，不能用同一把尺子衡量
- 文本 LLM 有系统性盲区，VLM 视觉校验必不可少
- 二值决策（keep/reject），不再有 maybe 灰区

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

## 7. 漏筛分析与后续规划

### 7.1 v4 改善的部分

与 v1-v3 相比，v4 Source Routing 缓解了以下问题：

| 问题 | v3 行为 | v4 改善 |
|------|---------|---------|
| Coarse 标注误杀 | hacs/thumos14 只有 1-3 标签，boundary/diversity 低 → reject | Group B 用 physical richness 评估，不看标注数量 |
| ASR 时间戳噪声 | 时间戳不对导致 boundary 低 → reject | Group C 忽略时间戳，只看文本内容 |
| 文本盲区 | 文本说"cooking"但视频可能是 talking head | VLM 视觉校验兜底 |

### 7.2 仍存在的局限

| 局限 | 影响 | 缓解方案 |
|------|------|---------|
| text_filter 的 ≥5 events 硬门槛 | 标注稀疏的好视频被丢弃 | 后续 DVC 重标注 |
| Group C 阈值较严（density >= 4） | ASR 样本通过率可能偏低 | 可根据实测调整 |
| VLM 依赖视频可访问性 | 需要 video root 路径 | 需确保计算节点能访问视频文件 |

### 7.3 后续计划：DVC 重标注

对文本粗筛 reject 但视频质量可能好的样本，计划引入 DVC（Dense Video Captioning）补救：

```
当前 pipeline                    DVC 补救路线
──────────                    ──────────────
text_filter → Stage A          被 reject 的子集
→ VLM Vision → keep           → VLM DVC 重标注
         │                    → 生成标准化事件
    最终候选池 ←── 合并 ──── 新候选
```

### 7.4 关于 decision_rules 的架构

| 数据源 | 规则位置 | 说明 |
|--------|---------|------|
| ET-Instruct | `et_instruct_164k/stage_a_coarse_filter.py` 内联 `apply_rules()` | per-group 阈值 |
| TimeLens | `shared/decision_rules.py:apply_richness_rules()` | 统一 boundary+diversity |
| VLM Vision | `shared/stage_a_vision_filter.py` 内联 `apply_vision_rules()` | score >= 3 + is_physical |
| Stage B | `shared/decision_rules.py:apply_stage_b_rules()` | overall + 3 维 |

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
│   │   ├── decision_rules.py        # 程序化决策规则 (TimeLens + Stage B)
│   │   ├── video_sampler.py         # 视频抽帧 (6帧, base64, 512px)
│   │   ├── stage_a_vision_filter.py # VLM 视觉校验 (通用)
│   │   ├── stage_b_fine_filter.py   # Stage B 层次潜力 (可选，当前不使用)
│   │   ├── analyze_results.py       # 结果统计分析 + HTML 报告
│   │   └── convert_to_viz.py        # 转可视化格式 + 抽帧
│   ├── et_instruct_164k/
│   │   ├── text_filter.py           # Step 0: 文本规则筛选
│   │   ├── stage_a_coarse_filter.py # Stage A: Source Routing (3 Groups)
│   │   ├── run_pipeline.sh          # 一键运行
│   │   └── results/                 # 输出目录
│   └── timelens_100k/
│       ├── text_filter.py
│       ├── stage_a_coarse_filter.py # Stage A: 边界清晰度 + 阶段多样性
│       ├── run_pipeline.sh
│       └── results/
```

---

## 9. 运行命令（从 train/ 目录）

### ET-Instruct Stage A（1K 抽样）

```bash
python proxy_data/data_curation/sources/et_instruct_164k/stage_a_coarse_filter.py \
    --input proxy_data/data_curation/sources/et_instruct_164k/results/passed.jsonl \
    --output proxy_data/data_curation/sources/et_instruct_164k/results/stage_a_results.jsonl \
    --sample-n 1000 --workers 16
```

### ET-Instruct Stage A（全量）

```bash
python proxy_data/data_curation/sources/et_instruct_164k/stage_a_coarse_filter.py \
    --input proxy_data/data_curation/sources/et_instruct_164k/results/passed.jsonl \
    --output proxy_data/data_curation/sources/et_instruct_164k/results/stage_a_results.jsonl \
    --no-sample --resume --workers 16
```

### VLM 视觉校验

```bash
python proxy_data/data_curation/sources/shared/stage_a_vision_filter.py \
    --input proxy_data/data_curation/sources/et_instruct_164k/results/stage_a_results_keep.jsonl \
    --output proxy_data/data_curation/sources/et_instruct_164k/results/vision_results.jsonl \
    --video-root /path/to/et_instruct/videos \
    --video-field video \
    --workers 4
```

### TimeLens Stage A

```bash
python proxy_data/data_curation/sources/timelens_100k/stage_a_coarse_filter.py \
    --input proxy_data/data_curation/sources/timelens_100k/results/passed_timelens.jsonl \
    --output proxy_data/data_curation/sources/timelens_100k/results/stage_a_results.jsonl \
    --sample-n 1000 --workers 16
```

### TimeLens VLM 视觉校验

```bash
python proxy_data/data_curation/sources/shared/stage_a_vision_filter.py \
    --input proxy_data/data_curation/sources/timelens_100k/results/stage_a_results_keep.jsonl \
    --output proxy_data/data_curation/sources/timelens_100k/results/vision_results.jsonl \
    --video-root /path/to/timelens/videos \
    --video-field video_path \
    --workers 4
```

### 查看结果

```bash
python proxy_data/data_curation/sources/shared/analyze_results.py \
    --input proxy_data/data_curation/sources/et_instruct_164k/results/stage_a_results.jsonl \
    --stage A --review 3 \
    --html proxy_data/data_curation/sources/et_instruct_164k/results/stage_a_report.html
```
