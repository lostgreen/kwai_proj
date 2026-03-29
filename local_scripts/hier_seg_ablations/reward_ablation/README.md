# Reward + Prompt Ablation 实验

层次分割 (Hierarchical Segmentation) 消融实验，对比不同 Reward 函数和 Prompt 设计对训练效果的影响。

## 实验概览

全部实验使用 **L2 + L3** 两层数据（L1 标注存在 warped 映射问题，暂不参与训练）。

### 实验矩阵

| 实验 | Prompt | Reward | 说明 |
|------|--------|--------|------|
| **R1** | V3 boundary V2 | F1-IoU (baseline) | Reward ablation: 匈牙利匹配 + F1 调和平均 |
| **R2** | V3 boundary V2 | Boundary-Aware | Reward ablation: 边界命中F1 + 段数准确性 + 覆盖率IoU |
| **PA1** | 原始标注 prompt | F1-IoU | Prompt ablation: cooking 领域词, 语义描述式 |
| **PA2** | V3 boundary V2 | F1-IoU | Prompt ablation: 边界判据 + 稀疏采样感知 + 硬规则 |

### 控制变量

- **模型**: Qwen3-VL-4B-Instruct
- **算法**: EMA-GRPO, LR=5e-7, cosine decay, 60 steps
- **数据**: L2+L3, 400 train + 100 val per level
- **视频**: 2 fps, max 256 frames

---

## Prompt 对比: 原始标注 vs V3 边界判据

### L2 层 — 事件检测

| 维度 | 原始 (PA1) | V3 边界判据 (PA2) |
|------|-----------|------------------|
| **定义** | "goal-directed cooking workflow that transforms ingredients" | "LOCAL TASK UNIT — coherent block accomplishing one self-contained local task" |
| **边界标准** | 未显式定义 | ✅ 显式: "cut when sub-goal achieved; DON'T cut when only tool changes" |
| **领域词汇** | "cooking", "recipe subgoal", "ingredients" | ✅ 无领域词 (domain-agnostic) |
| **稀疏采样** | 未提及 | ✅ "sampled at 1-2 fps, do NOT rely on single-frame micro-motions" |
| **硬规则** | 无 | ✅ min 5s, 2-8 units, 同 sub-goal 必须合并 |
| **CoT** | 无 (V1/V2), 有 (V3/V4) | 无 (V1/V2), 有 (V3/V4) |

### L3 层 — 原子操作 / 状态变化

| 维度 | 原始 (PA1) | V3 边界判据 (PA2) |
|------|-----------|------------------|
| **定义** | "atomic cooking actions (cutting, stirring, pouring)" | "VISIBLE STATE-CHANGE — shortest span where object undergoes clear, sustained change reliably inferred from sparse frames" |
| **核心差异** | 假设连续视频 → "atomic operation" 概念理想化 | ✅ 适配稀疏帧 → "minimal visible state-change segment" |
| **边界标准** | 隐式 | ✅ 显式: "cut when new object change begins/completes; DON'T cut for hand repositioning or single-frame flicker" |
| **硬规则** | 无 | ✅ min 2s, max 15s, 同对象同变化必须合并 |

### 设计哲学差异

```
原始 prompt (PA1):
  L2 = "cooking event"       → 模型按 "是什么" 分类
  L3 = "atomic cooking action" → 模型找 "动作名" 然后定位

V3 边界判据 (PA2):
  L2 = "local task unit"      → 模型按 "边界在哪" 切分
  L3 = "visible state-change" → 模型看 "状态是否变了" 然后定位

核心假设: 边界判据 > 语义描述 对低帧率视频更有效
```

---

## Reward 对比: F1-IoU vs Boundary-Aware

| 维度 | F1-IoU (R1) | Boundary-Aware (R2) |
|------|-------------|---------------------|
| **匹配策略** | 匈牙利最优匹配 (一对一) | 贪心最近边界匹配 |
| **核心指标** | F1 = 2·Recall·Precision / (R+P) | 0.5×BoundaryF1 + 0.2×CountAcc + 0.3×CoverageIoU |
| **段级 vs 边界级** | 段级匹配 — 每段 IoU 整体评估 | 边界级 — 分别评估起止点精度 |
| **段数敏感** | 通过 F1 隐式惩罚 | ✅ 显式高斯惩罚: exp(-Δn²/2σ²) |
| **覆盖率** | 隐含在 IoU 中 | ✅ 独立时间线 Coverage IoU |
| **容忍度** | IoU 严格按比例衰减 | 边界 τ=3s 内视为命中 |
| **预期行为** | 偏保守 (少输出, 高 IoU) | 鼓励输出正确数量 + 大致位置 |

---

## 用法

```bash
# 全部 4 组实验
bash run_reward_ablation.sh

# 仅 Reward Ablation
EXPS="R1 R2" bash run_reward_ablation.sh

# 仅 Prompt Ablation
EXPS="PA1 PA2" bash run_reward_ablation.sh

# 单个实验
EXPS="PA2" bash run_reward_ablation.sh

# 快速调试 (10 steps)
MAX_STEPS=10 EXPS="R1" bash run_reward_ablation.sh
```

---

## 文件结构

```
reward_ablation/
├── README.md                    # 本文件
├── run_reward_ablation.sh       # 批量运行入口
├── exp_r1_f1iou.sh             # Reward Ablation: F1-IoU baseline
├── exp_r2_boundary.sh          # Reward Ablation: Boundary-Aware
├── exp_pa1_original.sh         # Prompt Ablation: 原始标注 prompt
└── exp_pa2_v3boundary.sh       # Prompt Ablation: V3 边界判据 prompt

# 相关文件
verl/reward_function/
├── youcook2_hier_seg_reward.py           # F1-IoU reward (R1/PA1/PA2)
├── youcook2_hier_seg_reward_boundary.py  # Boundary-Aware reward (R2)
└── youcook2_temporal_seg_reward.py       # 共用工具 (IoU, NMS, Hungarian)

local_scripts/hier_seg_ablations/prompt_ablation/
├── prompt_variants_v2.py        # 原 V2 prompt 模板 (语义描述)
└── prompt_variants_v3.py        # V3 prompt 模板 (边界判据 + 稀疏采样)
```
