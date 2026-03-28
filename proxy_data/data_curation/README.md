# Data Curation — 数据筛选与候选构建

从多数据源中筛选适合进入 **层次分割标注 (Hierarchical Segmentation Annotation)** 的视频样本。

---

## 设计目标

1. **文本先行**：利用已有文本标注快速过滤，降低人工审查成本
2. **领域均衡**：跨 domain 控制样本比例，避免单一领域过拟合
3. **可扩展**：每个数据源独立目录，新增数据源只需添加一个子目录 + 配置
4. **可追溯**：每步过滤均保留 passed / rejected 列表及原因

---

## 目录结构

```
data_curation/
├── README.md                          ← 本文件
├── configs/                           ← 筛选配置（每数据源一份 YAML）
│   └── et_instruct_164k.yaml
│
├── sources/                           ← 各数据源筛选工作区
│   └── et_instruct_164k/             ← ET-Instruct-164K
│       ├── README.md                  ← 数据源基本信息 & 字段说明
│       ├── explore_data.py            ← 数据探索脚本（统计、抽样、格式检查）
│       ├── text_filter.py             ← 文本筛选主脚本
│       ├── results/                   ← 筛选产出
│       │   ├── raw_stats.json         ← 原始数据统计
│       │   ├── domain_stats.json      ← 各 domain 分布
│       │   ├── passed.jsonl           ← 通过筛选的样本
│       │   └── rejected.jsonl         ← 被拒样本（附 reason 字段）
│       └── sampled/                   ← 抽样审查子集
│           └── review_50.jsonl        ← 人工 review 样本
│
├── domain_balance/                    ← 跨数据源领域均衡分析
│   ├── analyze_balance.py             ← 均衡分析 & 采样脚本
│   └── balance_report.json            ← 最新均衡报告
│
└── output/                            ← 最终合并输出
    ├── merge_sources.py               ← 多数据源合并脚本
    └── candidates.jsonl               ← 进入层次分割标注的候选列表
```

---

## 筛选流水线

```
                          ┌─ Step 1: 元数据过滤 ─────────────────────────┐
[数据源 JSON]  ──►  text_filter.py                                       │
  163K 样本           │ 60s ≤ duration ≤ 240s                           │
                      │ events ≥ 5                                       │
                      │ dedup by video                                   │
                      │ domain cap ≤ 5000                                │
                      └──► passed.jsonl (~24K 视频)                      │
                                │                                        │
                      ┌─ Step 2: LLM 层次潜力评估 ──────────────────────┤
                      │  assess_hierarchy.py                             │
                      │  分析 L2 事件文本 →                              │
                      │    L1 聚合潜力 (events → phases?)                │
                      │    L3 分解潜力 (event → sub-actions?)            │
                      │    时序结构清晰度                                │
                      └──► assessed_high.jsonl (score ≥ 4)               │
                                │                                        │
                      ┌─ Step 3: 领域均衡 ──────────────────────────────┤
                      │  domain_balance/analyze_balance.py               │
                      └──► output/candidates.jsonl                       │
                                │                                        │
                      ──► 层次分割标注流水线 (annotation)                │
                                                                         │
                      rejected.jsonl (附 reason) ◄──────────────────────┘
```

### 层次映射关系

ET-Instruct 已有的事件标注 **= 我们的 L2**：

```
L1 (Macro Phases)    ←── LLM 评估：L2 事件能否聚合为 2-4 个阶段
                          例: [切菜, 备料, 调味] → "准备阶段"
                              [煎, 翻面, 调火] → "烹饪阶段"

L2 (Events)          ←── ET-Instruct 的 tgt 时间戳 + 事件描述（已有）
                          例: [36-44s] clean bananas
                              [49-57s] take skin off

L3 (Atomic Actions)  ←── LLM 评估：事件描述是否暗示可分解的子步骤
                          例: "clean bananas" → pick up / rinse / dry
```

---

## 文本筛选标准（初版，待根据数据格式调整）

### 适合层次分割的视频特征
- **时长**：60s–240s（足够 3 层分割，不会过长）
- **事件密度**：≥5 个可区分事件（保证 L1 聚合 + L3 分解空间）
- **L1 潜力**：事件可被分组为 2-4 个宏观阶段（如"准备→烹饪→摆盘"）
- **L3 潜力**：事件描述暗示可分解子步骤（非已是原子动作）
- **时序清晰**：事件间有明确先后关系，非随机/大量重叠

### 不适合的样本（排除）
- 纯对话/访谈类
- 静态画面或幻灯片式
- 标注过于简略（单句无结构）
- 时长 < 30s 或 > 600s

---

## 新增数据源流程

1. 在 `sources/` 下创建新目录：`sources/{source_name}/`
2. 复制并适配 `explore_data.py` 和 `text_filter.py`
3. 在 `configs/` 下新建 `{source_name}.yaml` 配置文件
4. 运行筛选，结果写入 `results/`
5. 运行 `domain_balance/analyze_balance.py` 更新均衡报告
6. 运行 `output/merge_sources.py` 生成最终候选

---

## 数据源列表

| 数据源 | 规模 | 视频域 | 状态 |
|--------|------|--------|------|
| ET-Instruct-164K | ~164K samples | ActivityNet, COIN, DiDeMo, Ego4D, HACS, HowTo, MR-HiSum, QuerYD, TACoS, ViTT | 🔄 筛选中 |
