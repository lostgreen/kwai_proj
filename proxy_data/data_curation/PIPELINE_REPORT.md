# LLM + VLM 数据筛选 Pipeline 汇报

> 文档目的：供 check 筛选流程、提示词设计、决策逻辑是否合理。
> 版本: v5 — Source Routing (4 Groups) + VLM Vision Filter

---

## 1. Pipeline 设计理念

### 1.1 为什么需要 Source Routing？

不同数据源的标注质量和粒度差异巨大：
- **Dense Manual**（coin, activitynet_captions 等）：有 5-20 个精细文本标注 → 可以判断边界清晰度和阶段多样性
- **Coarse Manual**（activitynet, hacs 等）：只有 1-3 个粗标签 → 只能判断活动本身是否物理丰富
- **ASR/Auto**（how_to_step, queryd 等）：时间戳不可靠 → 忽略时间戳，只看文本是否描述多步骤操作
- **VLM-Curated**（timelens 等）：标注由 VLM 精准生成，时间戳可靠 → 不审查标注质量，审查内容是否为多步骤物理过程

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
  │ Source      │                     │ Route D     │
  │ Routing     │                     │ (VLM-       │
  │ (A/B/C)    │                     │  Curated)   │
  └──────┬──────┘                     └──────┬──────┘
   keep │ reject                       keep │ reject
         │                              (pre-filter:
  ┌──────┴──────┐                      events<3→reject)
  │ VLM Vision  │                            │
  │ Filter      │                     ┌──────┴──────┐
  │ (6 frames)  │                     │ VLM Vision  │
  └──────┬──────┘                     │ Filter      │
   keep │ reject                      │ (6 frames)  │
         │                            └──────┬──────┘
    最终候选池                           keep │ reject
                                             │
                                        最终候选池
```

> **Stage B（层次潜力精筛）** 已移除。Route D + Vision Filter 已覆盖所需审查维度。

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

### 3.2 TimeLens — Route D (VLM-Curated)

TimeLens 的事件标注由 Gemini-2.5-Pro 精准生成，时间戳和事件描述高度可靠。因此 **不审查标注质量**，而审查：这些精准事件拼在一起，是否构成一个"物理层次丰富"的多步骤过程。

**Pre-filter（硬规则，跳过 LLM）**：
- `events < 3` → 直接 reject。如果 Gemini 也只能标出 1-2 个事件，说明视频极其单调。

**Group D Prompt**：

```json
{
  "process_analysis": "<1-2 sentences>",
  "physical_hierarchy_score": 1-5,
  "decision": "keep | reject"
}
```

核心判断：事件序列是否描述一个 **渐进式物理任务**（crafting, cooking, repairing, dynamic sports），而非新闻/采访/被动观察/单调重复。

**Reject 场景**：
- 新闻/采访/Vlog：人们坐着说话、切换镜头角度
- 被动观察：高速公路上的车流、睡觉的猫
- 单调/扁平：一个连续动作，无阶段划分

决策规则：`physical_hierarchy_score >= 3 → keep`，否则 `reject`。

### 3.3 程序化决策规则

规则在 `assess_sample()` 中**实时**应用：LLM 返回后立即检查，若与 LLM decision 不一致则覆写（原始决定存入 `_original_decision`）。

**ET-Instruct（per-group inline rules）**：
```python
Group A: phase_diversity_score >= 3          → keep
Group B: physical_richness_score >= 3        → keep
Group C: action_density_score >= 4           → keep
其余 / parse error                            → reject
```

**TimeLens Route D**（`decision_rules.py:apply_group_d_rules()`）：
```python
events < 3 (pre-filter)                             → reject（跳过 LLM）
physical_hierarchy_score >= 3                        → keep
其余 / parse error                                    → reject
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

## 4. 设计演化记录

Pipeline 经历了 5 个主要版本：

| 版本 | 设计 | 问题 |
|------|------|------|
| v1 | L2 粒度分类 (granularity_label) | 96% reject — 问错了问题 |
| v2 | 视频内容丰富度评估 (3 维 scores) | >90% keep — 太宽松 |
| v3 | 边界清晰度 + 阶段多样性 (2 维) | 留存率合理但不区分数据源特点 |
| v4 | Source Routing (ET-Instruct 3 组) + VLM Vision Filter | 按数据源特点评估 + 视觉安全网 |
| **v5 (当前)** | 4 组 Source Routing + Route D (VLM-Curated) | TimeLens 专属物理过程审查 + 预过滤 |

**v5 核心改进**：
- 新增 Group D (VLM-Curated)：TimeLens 标注由 Gemini 精准生成，无需审查标注质量，只审查物理内容丰富度
- Pre-filter：events < 3 直接 reject，连 LLM 都不用调（省钱省时间）
- TimeLens pipeline 集成 Vision Filter：`Stage A → Vision Filter → 最终候选`
- 二值决策（keep/reject）贯穿全流程

---

## 5. 开销估算

### ET-Instruct-164K

| 阶段 | 样本数 | 输入 tokens | 输出 tokens | 预估费用 | 时间(16w) |
|------|--------|-------------|-------------|----------|-----------|
| Stage A | ~10K | ~4.5M | ~1.2M | **~$17** | ~10min |

### TimeLens-100K

| 阶段 | 样本数 | 输入 tokens | 输出 tokens | 预估费用 | 时间(16w) |
|------|--------|-------------|-------------|----------|-----------|
| Stage A (Route D) | ~12K | ~5.4M | ~1.4M | **~$20** | ~12min |

### 总计

| | 样本数 | 费用 | 时间 |
|---|---|---|---|
| **Stage A 全量** | ~22K | ~$37 | ~22min |

*假设 Novita Gemini 2.5 Pro: ~$2/M input, ~$12/M output。实际可能偏差 ±30%。*
*VLM Vision Filter 费用另计（依赖 VLM 模型定价和样本数量）。*

---

## 6. 漏筛分析与后续规划

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
| ET-Instruct | `et_instruct_164k/stage_a_coarse_filter.py` 内联 `apply_rules()` | per-group 阈值 (A/B/C) |
| TimeLens | `shared/decision_rules.py:apply_group_d_rules()` | physical_hierarchy_score >= 3 |
| VLM Vision | `shared/stage_a_vision_filter.py` 内联 `apply_vision_rules()` | score >= 3 + is_physical |

---

## 7. 文件清单

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
│   │   ├── decision_rules.py        # 程序化决策规则 (Group D)
│   │   ├── video_sampler.py         # 视频抽帧 (6帧, base64, 512px)
│   │   ├── stage_a_vision_filter.py # VLM 视觉校验 (通用)
│   │   ├── analyze_results.py       # 结果统计分析 + HTML 报告
│   │   ├── visualize_distribution.py # Source/Duration 分布可视化
│   │   └── convert_to_viz.py        # 转可视化格式 + 抽帧
│   ├── et_instruct_164k/
│   │   ├── text_filter.py           # Step 0: 文本规则筛选
│   │   ├── stage_a_coarse_filter.py # Stage A: Source Routing (3 Groups)
│   │   ├── run_pipeline.sh          # 一键运行
│   │   └── results/                 # 输出目录
│   └── timelens_100k/
│       ├── text_filter.py
│       ├── stage_a_coarse_filter.py # Stage A: Route D (VLM-Curated 物理过程审查)
│       ├── run_pipeline.sh          # Stage A → Vision Filter → 最终候选
│       └── results/
```

---

## 8. 运行命令（从 train/ 目录）

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
    --video-root /m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/videos \
    --video-field video \
    --workers 4
```

> `--video-field video`：ET-Instruct 样本中 `"video": "coin/xxx.mp4"` 字段，与 `--video-root` 拼接成完整路径。

### TimeLens Stage A (Route D)

```bash
python proxy_data/data_curation/sources/timelens_100k/stage_a_coarse_filter.py \
    --input proxy_data/data_curation/sources/timelens_100k/results/passed_timelens.jsonl \
    --output proxy_data/data_curation/sources/timelens_100k/results/stage_a_results.jsonl \
    --sample-n 1000 --workers 16
```

> Route D 内置 pre-filter：events < 3 直接 reject，不调用 LLM。

### TimeLens VLM 视觉校验

```bash
python proxy_data/data_curation/sources/shared/stage_a_vision_filter.py \
    --input proxy_data/data_curation/sources/timelens_100k/results/stage_a_results_keep.jsonl \
    --output proxy_data/data_curation/sources/timelens_100k/results/vision_results.jsonl \
    --video-root /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/video_shards \
    --video-field video_path \
    --workers 4
```

> `--video-field video_path`：TimeLens 样本中 `"video_path": "cosmo_cap/xxx.mp4"` 字段。

### TimeLens 一键 Pipeline

```bash
cd proxy_data/data_curation/sources/timelens_100k

# 抽样试跑 (Route D 抽 200 条)
bash run_pipeline.sh --sample

# 全量: Stage A (Route D) → Vision Filter → 最终候选
bash run_pipeline.sh --full

# Stage A 已完成，只跑 Vision Filter
bash run_pipeline.sh --vision-only
```

> TimeLens 不使用 Stage B。Route D + Vision Filter 已覆盖物理过程审查和视觉质量两个维度。

### 查看结果

```bash
python proxy_data/data_curation/sources/shared/analyze_results.py \
    --input proxy_data/data_curation/sources/et_instruct_164k/results/stage_a_results.jsonl \
    --stage A --review 3 \
    --html proxy_data/data_curation/sources/et_instruct_164k/results/stage_a_report.html
```
