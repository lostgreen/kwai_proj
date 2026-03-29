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
├── configs/                           ← 筛选配置（每数据源一份 YAML）
│   ├── et_instruct_164k.yaml
│   └── timelens_100k.yaml
│
├── sources/                           ← 各数据源筛选工作区
│   ├── shared/                        ← 共享模块
│   │   ├── llm_client.py             ← LLM API 调用、JSON 解析、并发评估
│   │   ├── decision_rules.py         ← 程序化决策规则（覆盖 LLM decision）
│   │   └── stage_b_fine_filter.py    ← Stage B 精筛（数据源无关）
│   │
│   ├── et_instruct_164k/             ← ET-Instruct-164K
│   │   ├── text_filter.py             ← Step 0: 元数据过滤
│   │   ├── stage_a_coarse_filter.py   ← Step 1: L2 粒度粗筛
│   │   ├── assess_hierarchy.py        ← [legacy] 旧版单阶段评估
│   │   └── results/                   ← 筛选产出
│   │
│   └── timelens_100k/                ← TimeLens-100K
│       ├── text_filter.py             ← Step 0: 元数据过滤
│       ├── stage_a_coarse_filter.py   ← Step 1: L2 粒度粗筛
│       ├── assess_hierarchy.py        ← [legacy] 旧版单阶段评估
│       └── results/
│
├── domain_balance/                    ← 跨数据源领域均衡分析
│
└── output/                            ← 最终合并输出
    └── candidates.jsonl               ← 进入层次分割标注的候选列表
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
│  → stage_a_keep.jsonl (~进入 Stage B)                    │
│  → stage_a_maybe.jsonl (灰区，可人工复核)                 │
│  → stage_a_reject.jsonl (淘汰)                            │
└─────────────────────────────────────────────────────────┘
  │ (only keep)
  ▼
┌─────────────────────────────────────────────────────────┐
│ Stage B: 层次潜力精筛 — shared/stage_b_fine_filter.py     │
│                                                          │
│  核心问题：这些 L2-like 事件能否支撑完整的 3 层标注？      │
│  • L1 聚合潜力: 能否分组为宏观阶段？                      │
│  • L3 分解潜力: 事件内部能否拆出原子动作？                │
│  • 时序结构:    事件间有清晰先后关系？                     │
│                                                          │
│  输出字段：                                               │
│  • l1_potential:       1-5                                │
│  • l3_potential:       1-5                                │
│  • temporal_structure: 1-5                                │
│  • overall_score:      1-5                                │
│  • phase_sketch:       L1 分组草图                        │
│  • decision:           keep / maybe / reject             │
│                                                          │
│  决策规则：                                               │
│  keep   → overall≥4 + 所有维度≥3                         │
│  reject → overall≤2 or 任一维度≤1                         │
│  maybe  → 其他                                            │
│                                                          │
│  → stage_b_keep.jsonl (最终保留 → 标注流水线)             │
│  → stage_b_maybe.jsonl (边界样本 → 人工复核)              │
│  → stage_b_reject.jsonl (精筛淘汰)                        │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: 领域均衡 + 合并 → output/candidates.jsonl        │
└─────────────────────────────────────────────────────────┘
```

### 为什么要两阶段

单阶段评估会产生 **补偿效应**：当 LLM 同时被问"粒度对不对"、"能不能扩 L1"、"能不能拆 L3"、"整体适不适合"时，容易出现：

> 虽然当前粒度不准，但时序很清楚，又能想象出 phase grouping，于是 overall 还是给高分

两阶段设计：
- **Stage A** 只关注粒度 fit，不问扩展潜力 → 干净筛掉粒度不对的
- **Stage B** 只对粒度合格的样本评估潜力 → 不会被粒度问题干扰

---

### 层次映射关系

```
L1 (Macro Phases)    ←── Stage B 评估：L2 事件能否聚合为 2-4 个阶段

L2 (Events)          ←── 数据源已有标注（ET-Instruct tgt / TimeLens events）
                          Stage A 判断其粒度是否真的像 L2

L3 (Atomic Actions)  ←── Stage B 评估：事件描述是否暗示可分解的子步骤
```

---

## 程序化决策规则

`shared/decision_rules.py` 提供硬规则兜底，覆盖 LLM 不一致的 decision：

```bash
# Stage A 后校正
python shared/decision_rules.py \
    --input results/stage_a_results.jsonl \
    --output results/stage_a_ruled.jsonl \
    --stage A --override

# Stage B 后校正
python shared/decision_rules.py \
    --input results/stage_b_results.jsonl \
    --output results/stage_b_ruled.jsonl \
    --stage B --override
```

---

## 快速运行示例

```bash
# ET-Instruct-164K 完整 pipeline
cd sources/et_instruct_164k

# Step 0: 元数据过滤
python text_filter.py \
    --json_path /path/to/et_instruct_164k_txt.json \
    --output_dir results \
    --config ../../configs/et_instruct_164k.yaml

# Stage A: L2 粒度粗筛（先抽样 200 条看分布）
python stage_a_coarse_filter.py \
    --input results/passed.jsonl \
    --output results/stage_a_results.jsonl \
    --sample-n 200

# Stage A: 全量评估（断点续评）
python stage_a_coarse_filter.py \
    --input results/passed.jsonl \
    --output results/stage_a_results.jsonl \
    --no-sample --resume --workers 16

# Stage B: 精筛
python ../shared/stage_b_fine_filter.py \
    --input results/stage_a_results_keep.jsonl \
    --output results/stage_b_results.jsonl \
    --data-source et_instruct \
    --no-sample --workers 16

# 可选: 用硬规则校正 LLM decision
python ../shared/decision_rules.py \
    --input results/stage_b_results.jsonl \
    --output results/stage_b_ruled.jsonl \
    --stage B --override
```

---

## 新增数据源流程

1. 在 `sources/` 下创建新目录：`sources/{source_name}/`
2. 实现 `text_filter.py`（元数据过滤）
3. 实现 `stage_a_coarse_filter.py`（复用 Stage A prompt，适配事件解析）
4. 在 `shared/stage_b_fine_filter.py` 的 `PARSERS` 中注册事件解析函数
5. 在 `configs/` 下新建 `{source_name}.yaml` 配置文件
6. 运行完整 pipeline，结果写入 `results/`

---

## 数据源列表

| 数据源 | 规模 | 视频域 | 状态 |
|--------|------|--------|------|
| ET-Instruct-164K | ~164K samples | ActivityNet, COIN, DiDeMo, Ego4D, HACS, HowTo, MR-HiSum, QuerYD, TACoS, ViTT | 两阶段筛选中 |
| TimeLens-100K | ~100K samples | cosmo_cap, internvid_vtime, didemo, queryd, hirest_step, hirest_grounding, hirest | 两阶段筛选中 |
