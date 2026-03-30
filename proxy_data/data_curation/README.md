# Data Curation — 数据筛选与候选构建

从多数据源中筛选适合进入 **层次分割标注 (Hierarchical Segmentation Annotation)** 的视频样本。

---

## 设计目标

1. **文本先行**：利用已有文本标注快速过滤，降低人工审查成本
2. **两阶段 LLM 筛选**：先粗筛粒度，再精筛潜力，避免一步到位的补偿效应
3. **领域均衡**：跨 domain 控制样本比例，避免单一领域过拟合
4. **保守偏置**：误筛进来 > 错杀，拿不准时不保留
5. **可扩展**：每个数据源独立目录，新增数据源只需添加一个子目录 + 配置
6. **可追溯**：每步过滤均保留 passed / rejected 列表及原因

---

## 目录结构

```
data_curation/
├── README.md                          ← 本文件
├── PIPELINE_REPORT.md                 ← Pipeline 设计汇报
├── merge_candidates.py                ← 多数据源合并脚本
│
├── configs/                           ← 筛选配置（每数据源一份 YAML）
│   ├── et_instruct_164k.yaml
│   └── timelens_100k.yaml
│
├── shared/                            ← 共享代码模块
│   ├── llm_client.py                 ← LLM API 调用、JSON 解析、并发评估
│   ├── decision_rules.py             ← 程序化决策规则（覆盖 LLM decision）
│   ├── vision_filter.py              ← VLM 视觉校验（数据源无关）
│   ├── video_sampler.py              ← 视频抽帧工具
│   ├── analyze_results.py            ← 评估结果分析与可视化
│   ├── convert_to_viz.py             ← 转换为标注可视化格式
│   └── visualize_distribution.py     ← 数据分布可视化（source/duration）
│
├── et_instruct_164k/                  ← ET-Instruct-164K 数据源脚本
│   ├── text_filter.py                ← Step 0: 元数据过滤
│   ├── stage_a_coarse_filter.py      ← Stage A: Source Routing 粗筛
│   ├── explore_data.py               ← 数据探索
│   └── run_pipeline.sh               ← 一键运行脚本
│
├── timelens_100k/                     ← TimeLens-100K 数据源脚本
│   ├── text_filter.py                ← Step 0: 元数据过滤
│   ├── stage_a_coarse_filter.py      ← Stage A: Route D 物理过程审查
│   └── run_pipeline.sh               ← 一键运行脚本
│
└── results/                           ← 所有数据产出（代码与数据分离）
    ├── et_instruct_164k/
    │   ├── vision_results_keep.jsonl ← ET-Instruct 最终候选 (4746 条)
    │   └── figures/                  ← 分布可视化图表
    ├── timelens_100k/
    │   ├── stage_a_results_keep.jsonl ← TimeLens Stage A 候选 (3002 条)
    │   └── figures/
    └── merged/
        ├── candidates.jsonl          ← 合并后最终候选 (7748 条)
        └── figures/
```

---

## 三步筛选流水线

```
[数据源 JSON]
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 0: 元数据过滤 — text_filter.py                      │
│  • 60s ≤ duration ≤ 240s                                 │
│  • events ≥ 5                                            │
│  • dedup by video                                        │
│  • domain cap                                            │
│  → passed.jsonl                                          │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Stage A: L2 粒度粗筛 — stage_a_coarse_filter.py          │
│                                                          │
│  核心问题：这些事件标注的粒度像不像 L2？                  │
│  不评估潜力，只判断粒度 fit                               │
│                                                          │
│  输出字段：                                               │
│  • granularity_label:  mostly_L2_like / L1 / L3 / mixed │
│  • l2_fit_score:       1-5                                │
│  • granularity_issue:  too_coarse / good / too_fine / mixed │
│  • mixed_ratio_estimate: low / medium / high             │
│  • decision:           keep / maybe / reject             │
│                                                          │
│  决策规则：                                               │
│  keep   → mostly_L2_like + score≥4 + good + low mixed   │
│  reject → mostly_L1/L3 or score≤2                        │
│  maybe  → 其他                                            │
│                                                          │
│  → stage_a_keep.jsonl (~进入 Stage B 或 Vision Filter)   │
│  → stage_a_maybe.jsonl (灰区，可人工复核)                 │
│  → stage_a_reject.jsonl (淘汰)                            │
└─────────────────────────────────────────────────────────┘
  │ (only keep)
  ▼
┌─────────────────────────────────────────────────────────┐
│ Vision Filter: VLM 视觉校验 — shared/vision_filter.py    │
│                                                          │
│  对 Stage A keep 的样本用 VLM 查看实际视频帧             │
│  过滤 talking head / 静态场景 / 游戏动画等                │
│                                                          │
│  → vision_results_keep.jsonl (最终保留 → 标注流水线)     │
│  → vision_results_reject.jsonl (视觉淘汰)                │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: 合并 → results/merged/candidates.jsonl           │
│  python merge_candidates.py                              │
└─────────────────────────────────────────────────────────┘
```

### 为什么要两阶段

单阶段评估会产生 **补偿效应**：当 LLM 同时被问"粒度对不对"、"能不能扩 L1"、"能不能拆 L3"、"整体适不适合"时，容易出现：

> 虽然当前粒度不准，但时序很清楚，又能想象出 phase grouping，于是 overall 还是给高分

两阶段设计：
- **Stage A** 只关注粒度 fit，不问扩展潜力 → 干净筛掉粒度不对的
- **Stage B / Vision** 只对粒度合格的样本做后续校验 → 不会被粒度问题干扰

---

### 层次映射关系

```
L1 (Macro Phases)    ←── 下游标注评估：L2 事件能否聚合为 2-4 个阶段

L2 (Events)          ←── 数据源已有标注（ET-Instruct tgt / TimeLens events）
                          Stage A 判断其粒度是否真的像 L2

L3 (Atomic Actions)  ←── 下游标注评估：事件描述是否暗示可分解的子步骤
```

---

## 程序化决策规则

`shared/decision_rules.py` 提供硬规则兜底，覆盖 LLM 不一致的 decision：

```bash
# Stage A 后校正
python shared/decision_rules.py \
    --input results/et_instruct_164k/stage_a_results.jsonl \
    --output results/et_instruct_164k/stage_a_ruled.jsonl \
    --stage A --override
```

---

## 快速运行示例

```bash
# ET-Instruct-164K 完整 pipeline
cd et_instruct_164k

# Step 0: 元数据过滤
python text_filter.py \
    --json_path /path/to/et_instruct_164k_txt.json \
    --output_dir ../results/et_instruct_164k \
    --config ../configs/et_instruct_164k.yaml

# Stage A: L2 粒度粗筛（先抽样 200 条看分布）
python stage_a_coarse_filter.py \
    --input ../results/et_instruct_164k/passed.jsonl \
    --output ../results/et_instruct_164k/stage_a_results.jsonl \
    --sample-n 200

# Stage A: 全量评估（断点续评）
python stage_a_coarse_filter.py \
    --input ../results/et_instruct_164k/passed.jsonl \
    --output ../results/et_instruct_164k/stage_a_results.jsonl \
    --no-sample --resume --workers 16

# Vision Filter: VLM 视觉校验
python ../shared/vision_filter.py \
    --input ../results/et_instruct_164k/stage_a_results_keep.jsonl \
    --output ../results/et_instruct_164k/vision_results.jsonl \
    --video-root /path/to/videos \
    --video-field video --workers 4

# 合并多数据源
cd ..
python merge_candidates.py \
    --inputs results/et_instruct_164k/vision_results_keep.jsonl \
            results/timelens_100k/stage_a_results_keep.jsonl \
    --outdir results/merged

# 可视化分布
python shared/visualize_distribution.py \
    --input results/merged/candidates.jsonl \
    --outdir results/merged/figures
```

---

## 新增数据源流程

1. 创建数据源目录：`{source_name}/`
2. 实现 `text_filter.py`（元数据过滤）
3. 实现 `stage_a_coarse_filter.py`（复用 Stage A prompt，适配事件解析）
4. 在 `configs/` 下新建 `{source_name}.yaml` 配置文件
5. 运行完整 pipeline，结果写入 `results/{source_name}/`
6. 更新 `merge_candidates.py` 添加新数据源的解析器

---

## 数据源列表

| 数据源 | 规模 | 视频域 | 最终候选 |
|--------|------|--------|----------|
| ET-Instruct-164K | ~164K samples | ActivityNet, COIN, DiDeMo, Ego4D, HACS, HowTo, MR-HiSum, QuerYD, TACoS, ViTT | 4746 条 |
| TimeLens-100K | ~100K samples | cosmo_cap, internvid_vtime, didemo, queryd, hirest_step, hirest_grounding, hirest | 3002 条 |
| **合并总计** | | 14 个 source | **7748 条** |
