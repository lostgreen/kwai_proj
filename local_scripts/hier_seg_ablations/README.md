# 三层分割 (Hier Seg) 消融实验

基于 YouCook2 分层时序标注（L1 宏观阶段 / L2 事件检测 / L3 原子动作定位），通过 RL 训练消融验证最优层级组合。

## 研究问题

**哪些层级的组合最能通过 RL 训练提升时序分割能力？** L3 的 query 顺序（正序/乱序）对训练有何影响？

## 三层任务设计

| 层级 | problem_type | 粒度 | 输入 | 输出 | Reward | 样本数 |
|------|-------------|------|------|------|--------|--------|
| L1 | `temporal_seg_hier_L1` | 阶段级 | warped 256帧合成视频 | 帧号区间 3-5 段 | F1-IoU (NMS+Hungarian) | 500 |
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

L3 seg 数据需要先在服务器上运行 `run_build.sh`（新增了 `--level 3s` 构建步骤）生成 `youcook2_hier_L3_seg_train_clipped.jsonl`。

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
├── README.md               # 本文件
├── common.sh               # 共用超参数
├── prepare_data.py          # 数据准备（按层/变体筛选 + split + merge）
├── launch_train.sh          # 统一训练入口
├── exp1_L2_only.sh
├── exp2_L3_seq.sh
├── exp3_L3_shuf.sh
├── exp4_L3_both.sh
├── exp5_L2_L3.sh
├── exp6_L1_L2_L3.sh
└── exp7_all_mixed.sh

proxy_data/youcook2_seg_annotation/datasets/
├── youcook2_hier_L1_train_clipped.jsonl       # 500 samples
├── youcook2_hier_L2_train_clipped.jsonl       # 1898 samples
├── youcook2_hier_L3_train_clipped.jsonl       # 3568 samples (grounding: seq+shuf)
├── youcook2_hier_L3_seg_train_clipped.jsonl   # ~376 samples (segmentation: 无 query)
└── ablation_data/
    ├── hier_seg_exp1_L2_only/             # 自动生成
    │   ├── train.jsonl
    │   └── val.jsonl
    └── ...

verl/reward_function/
└── youcook2_hier_seg_reward.py   # L1/L2: F1-IoU, L3: aligned tIoU
```
