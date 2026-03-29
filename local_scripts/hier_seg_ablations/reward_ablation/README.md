# Reward Ablation — 层次分割 Reward 消融实验

> 核心问题：**不同 Reward 信号如何影响 RL 训练的时序分割质量？**
>
> 两组实验 (R1/R2) 固定 Prompt (V3 V2 变体)，只改变 Reward 函数。

---

## 消融目的

RL 训练中 Reward 信号直接决定策略优化方向。对于时序分割任务，一个关键问题是：

**Reward 应该关注「段的整体匹配质量」还是「边界定位精度 + 段数正确性」？**

- **段级匹配** (F1-IoU)：通过 IoU 衡量 pred 与 GT 段的重叠度，整体评价较为保守
- **边界级评估** (Boundary-Aware)：将段分解为起止点，分别评估边界精度、段数、时间覆盖

两种 Reward 给模型的训练信号不同，可能导致截然不同的行为模式：
- F1-IoU 可能鼓励模型**少输出、但输出的段 IoU 高**（保守策略）
- Boundary-Aware 可能鼓励模型**输出正确数量的段、边界大致正确**（平衡策略）

消融实验旨在量化这两种信号的差异，为后续实验选择最优 Reward 提供依据。

---

## 实验设计

### 两组对比

| 实验 | Reward 函数 | 文件 |
|------|------------|------|
| **R1** | F1-IoU (baseline) | `youcook2_hier_seg_reward.py` |
| **R2** | Boundary-Aware | `youcook2_hier_seg_reward_boundary.py` |

### 控制变量

| 维度 | 值 |
|------|-----|
| Prompt | V3 边界判据 V2 变体（两组共用） |
| 模型 | Qwen3-VL-4B-Instruct |
| 数据层 | L2 + L3（跳过 L1） |
| 算法 | EMA-GRPO, LR=5e-7, cosine decay |
| 训练 | 60 steps, rollout_bs=16, 8 GPU |
| 数据量 | 每层 400 train + 100 val |

---

## Reward 函数详细设计

### R1: F1-IoU (Baseline)

**核心思想**: 段级最优匹配 — 每个 pred 段找到最匹配的 GT 段，计算整体 F1。

```
Pipeline:
  pred_segs, gt_segs → NMS去重 → 匈牙利最优匹配(IoU矩阵) → F1 = 2·P·R / (P+R)

匹配逻辑:
  1. 对 pred 和 gt 分别做 NMS (阈值 0.3)，去除高度重叠的段
  2. 构建 |pred| × |gt| 的 IoU 矩阵
  3. 匈牙利算法求一对一最优匹配 (最大化总IoU)
  4. 匹配对中 IoU > 阈值(0.3) 视为 TP
  5. F1 = 2 × Precision × Recall / (P + R)
```

**设计理由**:
- 匈牙利匹配保证**全局最优对齐**，不受 pred/gt 排序影响
- 段级 IoU 评估整段重叠度，对断裂/合并预测有较好的容错性
- 是时序分割领域的标准评估方式，可与已有 benchmark 直接对比

**潜在局限**:
- IoU 对段长度敏感：相同位移偏差，短段 IoU 衰减比长段快
- 对段数的约束只通过 F1 隐式施加（多输出时 Precision 降低）
- 保守行为：模型可能学到"少输出、但输出的段尽量精确"的策略

### R2: Boundary-Aware

**核心思想**: 将段分解为边界点，分三个维度独立评估，加权求和。

```
overall = 0.5 × boundary_f1 + 0.2 × count_accuracy + 0.3 × coverage_iou
```

#### 组件 1: Boundary Hit F1 (权重 0.5)

```
Pipeline:
  pred_segs → 提取所有 start/end → 排序
  gt_segs   → 提取所有 start/end → 排序
  
  贪心匹配: 距离矩阵 → 按距离升序 → 每个边界只匹配一次
  命中条件: |pred_boundary - gt_boundary| ≤ τ (τ=3秒)
  
  recall = hits / n_gt_boundaries
  precision = hits / n_pred_boundaries
  F1 = 2 × P × R / (P + R)
```

**设计理由**:
- τ=3s 容忍度对应 2fps 采样下的 6 帧窗口，允许合理的对齐误差
- 贪心匹配比匈牙利计算轻量，且在边界点密集时更鲁棒
- 独立评估起止点精度，比段级 IoU 更精细

#### 组件 2: Count Accuracy (权重 0.2)

```
count_acc = exp(-(n_pred - n_gt)² / (2 × σ²))    σ=2.0

n_pred = n_gt:    score = 1.0
差 1 段:           score ≈ 0.88
差 2 段:           score ≈ 0.61
差 3 段:           score ≈ 0.32
差 4 段:           score ≈ 0.14
```

**设计理由**:
- **显式惩罚段数偏差**：F1-IoU 只通过 Precision/Recall 隐式约束，信号弱
- Gaussian 衰减平滑：接近正确段数时梯度小（容忍±1），偏差大时梯度急（强惩罚）
- σ=2.0 允许 ±1 段容忍（score > 0.88），对分割粒度不确定性友好

#### 组件 3: Coverage IoU (权重 0.3)

```
Pipeline:
  pred_segs → 合并重叠区间 → pred_mask (时间线)
  gt_segs   → 合并重叠区间 → gt_mask (时间线)
  
  intersection = 双指针法计算交集长度
  union = len(pred_mask) + len(gt_mask) - intersection
  coverage_iou = intersection / union
```

**设计理由**:
- 确保 pred 和 gt 覆盖的**总时间范围**一致
- 即使边界位移了几秒、段数不完全正确，只要覆盖区域对就有分
- 防止模型只输出几个高精度短段而忽略大量时间区域

### 权重设计

| 组件 | 权重 | 理由 |
|------|------|------|
| Boundary F1 | 0.5 | **主信号** — 边界位置是分割任务的核心 |
| Coverage IoU | 0.3 | **辅助** — 保证不遗漏大段时间区域 |
| Count Accuracy | 0.2 | **约束** — 防止过碎/过粗分割 |

---

## 预期行为对比

| 维度 | F1-IoU (R1) | Boundary-Aware (R2) |
|------|-------------|---------------------|
| 段数倾向 | 偏少（少输出保证精度） | 接近 GT 数量（count 显式约束） |
| 边界精度 | 间接约束（通过 IoU） | 直接约束（τ=3s 命中判定） |
| 对短段 | 不友好（短段 IoU 衰减快） | 友好（边界τ容忍度固定） |
| 对遗漏 | 通过 Recall 惩罚 | 通过 Recall + Coverage 双重惩罚 |
| 计算复杂度 | O(n²) 匈牙利 | O(n²) 贪心（实际更快） |

---

## 用法

```bash
# 两组 Reward Ablation
bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh

# 仅 R1 baseline
EXPS="R1" bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh

# 仅 R2 boundary
EXPS="R2" bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh

# 快速调试
MAX_STEPS=10 EXPS="R1" bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh
```

---

## 文件结构

```
reward_ablation/
├── README.md                    # 本文件
├── run_reward_ablation.sh       # 批量运行入口 (R1/R2)
├── exp_r1_f1iou.sh             # R1: F1-IoU baseline
└── exp_r2_boundary.sh          # R2: Boundary-Aware

# Reward 函数实现
verl/reward_function/
├── youcook2_hier_seg_reward.py           # F1-IoU (R1)
├── youcook2_hier_seg_reward_boundary.py  # Boundary-Aware (R2)
└── youcook2_temporal_seg_reward.py       # 共用工具 (parse, IoU, NMS, Hungarian)
```
