# Chain-Seg 消融实验 (Track 4)

链式层次分割的**任务形式**消融：对比两种不同的 L2/L3 组合方式。

## 消融变体

| Variant | L2 任务 | L3 任务 | 输入 | problem_type |
|---------|---------|---------|------|-------------|
| **V1 (dual-seg)** | 自由分割 | 自由分割 | 128s 视频 | `temporal_seg_chain_dual_seg` |
| **V2 (ground-seg)** | 单 caption grounding | 自由分割 | 128s 视频 + 1 条 caption | `temporal_seg_chain_ground_seg` |

## 目录结构

```
chain_seg_ablation/
├── README.md                         # 本文件
├── prompt_variants_chain.py          # V1/V2 prompt 模板
├── prepare_chain_ablation_data.py    # 数据准备
├── exp_chain_ablation.sh             # 单次实验入口
└── run_chain_ablation.sh             # 批量运行
```

## Reward 设计

共享文件: `verl/reward_function/youcook2_chain_seg_reward.py`

公式: `R = 0.4 × R_L2 + 0.6 × R_L3 × max(R_L2, 0.3)`

| | V1 (dual-seg) | V2 (ground-seg) |
|---|---|---|
| L2 评估 | F1-IoU (Hungarian 匹配) | temporal_iou (单段) |
| L3 评估 | 按 matched L2 配对 → F1-IoU | clip 到 L2 bounds → F1-IoU |
| L3 边界约束 | clip L3 到 matched L2 段内 | clip L3 到 predicted L2 段内 |

## 运行

```bash
# 单个变体
VARIANT=V1 bash local_scripts/hier_seg_ablations/chain_seg_ablation/exp_chain_ablation.sh

# 批量 V1 → V2
bash local_scripts/hier_seg_ablations/chain_seg_ablation/run_chain_ablation.sh

# 仅准备数据
python3 local_scripts/hier_seg_ablations/chain_seg_ablation/prepare_chain_ablation_data.py \
  --variant V1 --output-dir /tmp/chain_seg_V1

# 自定义路径
ABLATION_DATA_ROOT=/custom/path HIER_DATA_ROOT=/custom/src \
  VARIANT=V2 bash local_scripts/hier_seg_ablations/chain_seg_ablation/exp_chain_ablation.sh
```

## 数据量

- V1: ~737 窗口 (同 exp8, `--min-events 2`)，每窗口多事件
- V2: 拆单事件后 ~1800+ 样本 (每 matched event 一条)

## 测试

```bash
python3 tests/test_chain_seg_reward.py
```
