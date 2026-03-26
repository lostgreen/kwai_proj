# Event Logic 消融实验

## 实验设计

纯 Event Logic 消融，基于 L2 分层标注（管线 2），不混合 AoT 数据。

### 实验矩阵

| 实验 | 脚本 | 任务组合 | add | replace | sort | AI 过滤 | 验证目标 |
|------|------|---------|-----|---------|------|---------|---------|
| exp1 | `exp1_add_only.sh` | Add Only | 3/video | 0 | 0 | - | 单任务：因果预测 |
| exp2 | `exp2_replace_only.sh` | Replace Only | 0 | 3/video | 0 | - | 单任务：缺失补全 |
| exp3 | `exp3_sort_only.sh` | Sort Only | 0 | 0 | 3/video | - | 单任务：时序排序 |
| exp4 | `exp4_add_replace.sh` | Add + Replace | 2/video | 2/video | 0 | - | MCQ 双任务联合 |
| exp5 | `exp5_all_mixed.sh` | All Mixed | 2/video | 2/video | 1/video | - | 三任务全混合 |
| exp6 | `exp6_all_filtered.sh` | All + Filter | 2/video | 2/video | 1/video | ON | AI 过滤效果 |

### 机器分配

```
group=1 (Machine A): exp1 → exp4
group=2 (Machine B): exp2 → exp5
group=3 (Machine C): exp3 → exp6
```

## 运行方式

```bash
# 单实验
MAX_STEPS=60 bash local_scripts/event_logic_ablations/exp1_add_only.sh

# 批量（分配到 3 台机器）
MAX_STEPS=60 bash local_scripts/event_logic_ablations/run_batch.sh 1  # Machine A
MAX_STEPS=60 bash local_scripts/event_logic_ablations/run_batch.sh 2  # Machine B
MAX_STEPS=60 bash local_scripts/event_logic_ablations/run_batch.sh 3  # Machine C
```

## 数据流

```
L2 annotations (youcook2_seg_annotation/annotations/*.json)
    │
    │  build_l2_event_logic.py
    │  --add-per-video N --replace-per-video N --sort-per-video N
    │  [--filter]  ← exp6 启用 AI 因果过滤
    │
    ▼
l2_event_logic_raw.jsonl
    │
    │  shuffle + split 5% val
    │
    ├─→ mixed_train.jsonl
    │       │
    │       │  offline_rollout_filter.py（多卡并行）
    │       │
    │       ▼
    │   mixed_train.offline_filtered.jsonl
    │       │
    │       │  [curate_1k_samples.py]  ← SKIP_CURATE=true 时跳过
    │       │
    │       ▼
    │   TRAIN_FILE → verl.trainer.main
    │
    └─→ mixed_val.jsonl → TEST_FILE
```

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| MIN_EVENTS | 4 | 视频最少事件数 |
| MIN_CONTEXT / MAX_CONTEXT | 2 / 4 | add 任务上下文长度范围 |
| REPLACE_SEQ_LEN | 5 | replace 任务序列长度 |
| SORT_SEQ_LEN | 5 | sort 任务序列长度 |
| FILTER_AI | false | 是否启用 VLM 因果过滤 |
| SKIP_CURATE | true | 跳过难度采样（用在线过滤代替） |
| ONLINE_FILTERING | true | 训练时在线过滤 |

## 文件结构

```
event_logic_ablations/
├── README.md              ← 本文件
├── common.sh              ← 共用超参数
├── launch_train.sh        ← 完整管线（构造→切分→筛选→训练）
├── run_batch.sh           ← 批量运行器（分 3 组）
├── exp1_add_only.sh       ← 实验 1: Add Only
├── exp2_replace_only.sh   ← 实验 2: Replace Only
├── exp3_sort_only.sh      ← 实验 3: Sort Only
├── exp4_add_replace.sh    ← 实验 4: Add + Replace
├── exp5_all_mixed.sh      ← 实验 5: All Mixed
└── exp6_all_filtered.sh   ← 实验 6: All + AI Filter
```

## 与 AoT 消融的区别

| | AoT 消融 | Event Logic 消融 |
|---|---|---|
| 数据源 | 数据源 C（temporal_aot） | 数据源 B（L2 分层标注） |
| 任务类型 | V2T/T2V + 2-way/4-way | add/replace/sort |
| 构造脚本 | `build_aot_mcq.py` | `build_l2_event_logic.py` |
| 负例来源 | 同 clip 的方向变体 caption | 同菜谱/跨菜谱 instruction |
| Reward | MCQ 精确匹配 | MCQ 精确匹配 + Jigsaw Displacement |
| 答案重平衡 | 需要（binary A/B 偏斜） | 不需要（4-way ABCD 天然均衡） |
