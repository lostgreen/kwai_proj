# 三层分割 (Hier Seg) 训练 & 消融实验

基于分层时序标注（L1 宏观阶段 / L2 事件检测 / L3 原子动作分割），使用 NGIoU V2 reward 训练。

---

## 快速开始 — 三层训练 (推荐)

### 1. 构建数据

```bash
# 一键构建 L1+L2+L3 训练/验证数据
bash proxy_data/youcook2_seg/hier_seg_annotation/build_v1_data.sh

# 带 hint 版本
bash proxy_data/youcook2_seg/hier_seg_annotation/build_v1_data.sh --use-hint
```

### 2. 启动训练

```bash
# 默认：8卡, L1+L2+L3, NGIoU reward, 60 steps
bash local_scripts/hier_seg_ablations/train_hier_seg.sh

# 自定义
MAX_STEPS=30 EXP_NAME=my_exp bash local_scripts/hier_seg_ablations/train_hier_seg.sh

# 使用 hint 数据
USE_HINT=true bash local_scripts/hier_seg_ablations/train_hier_seg.sh

# 指定数据目录
DATA_DIR=/path/to/data bash local_scripts/hier_seg_ablations/train_hier_seg.sh
```

### 3. Baseline 评估

```bash
python local_scripts/hier_seg_ablations/eval_baseline_rollout.py \
    --input-dir /path/to/train/ \
    --model-path /home/xuboshen/models/Qwen3-VL-4B-Instruct \
    --sample-per-level 50 \
    --use-hint \
    --output-dir ./eval_results/
```

---

## Reward V2: NGIoU 分层奖励

详见 [`verl/reward_function/REWARD_DESIGN_V2.md`](../../verl/reward_function/REWARD_DESIGN_V2.md)

| 层级 | 匹配 | 度量 | 特殊机制 |
|------|------|------|---------|
| L1 宏观阶段 | Hungarian | Margin-Relaxed F1-NGIoU | GT ±5s 宽容 |
| L2 中观事件 | Hungarian | F1-NGIoU | 标准 |
| L3 原子动作 | Hungarian | F1-NGIoU | 无 NMS |

NGIoU 核心优势：pred 与 GT 不重叠时仍有非零信号（越近越高），为 RL 提供连续梯度。

---

## 共用超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Model | Qwen3-VL-4B-Instruct | |
| Algorithm | `ema_grpo` | |
| LR | 5e-7 | cosine warmup/decay |
| Online filtering | 启用 | |
| Rollout N | 8 | |
| Max steps | 60 | |
| TP size | 2 | |
| GPUs | 8 | |

完整参数见 [common.sh](common.sh)。

---

## 文件结构

```
local_scripts/hier_seg_ablations/
├── README.md                          # 本文件
├── common.sh                          # 共用超参数
├── launch_train.sh                    # 统一训练入口
├── train_hier_seg.sh                  # ← 三层训练主脚本 (L1+L2+L3, NGIoU)
├── build_hier_data.py                 # → proxy_data/.../build_hier_data.py (symlink)
├── eval_baseline_rollout.py           # Baseline rollout 评估
│
├── prompt_ablation/                   # Track 2: Prompt 消融 (PA1/PA2)
│   ├── exp_pa1_original.sh            #   烹饪域原始 prompt
│   ├── exp_pa2_v3boundary.sh          #   领域无关 + 边界判据 prompt
│   ├── prepare_prompt_data.py         #   Prompt variant 替换
│   └── prompt_variants_v3.py          #   V1-V4 prompt 模板
│
├── reward_ablation/                   # Track 3: Reward 消融 (R1/R2)
│   ├── exp_r1_f1iou.sh               #   R1: F1-IoU (baseline)
│   └── exp_r2_boundary.sh            #   R2: Boundary-Aware reward
│
└── chain_seg_ablation/                # Track 4: Chain-of-Segment
    ├── build_chain_seg_data.py
    └── exp_chain_ablation.sh

verl/reward_function/
├── youcook2_hier_seg_reward.py        # V2: NGIoU 分层 reward (L1/L2/L3)
├── youcook2_temporal_seg_reward.py    # 基础: ngiou(), compute_f1_ngiou()
├── youcook2_chain_seg_reward.py       # Chain-Seg: 0.4*tIoU + 0.6*F1-IoU
├── youcook2_hier_seg_reward_boundary.py  # Boundary-Aware reward (消融用)
└── REWARD_DESIGN_V2.md                # NGIoU 设计文档
```

---

## 数据流程

```
原始标注 JSON (annotations/*.json)
    ↓ build_v1_data.sh
    │ ├ build_hier_data.py (per-phase L2, 筛选, 均衡采样)
    │ └ prepare_clips.py (L1@1fps, L2@2fps, L3@2fps)
    ↓
train_all.jsonl / val_all.jsonl (L1+L2+L3 合并)
    ↓ train_hier_seg.sh
EasyR1 训练 (youcook2_hier_seg_reward → NGIoU V2)
    ↓
Checkpoint / TensorBoard
```

---

## 消融实验

### Prompt 消融 (prompt_ablation/)

| 实验 | Prompt 类型 | 说明 |
|------|------------|------|
| PA1 | 原始烹饪域 prompt | 基线 |
| PA2 | V3 边界判据 + 领域无关 | 推荐 |

### Reward 消融 (reward_ablation/)

| 实验 | Reward | 说明 |
|------|--------|------|
| R1 | F1-IoU | 基线 (V1) |
| R2 | Boundary-Aware | 边界精度+计数 |

### Chain-Seg 消融 (chain_seg_ablation/)

| 实验 | 结构 | 说明 |
|------|------|------|
| chain | L2 grounding → L3 seg | 先粗后精链式推理 |
