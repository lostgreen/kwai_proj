# Chain-Seg 消融实验 (Track 4)

链式层次分割的**任务形式**消融：对比三种不同的 L2/L3 组合方式。

## 数据来源改进

旧方案从中间 L2/L3 JSONL 文件间接匹配，存在以下问题：
- L2 caption 绕道 L3 metadata 取得（`action_query` 字段实际是 L2 instruction）
- 跨窗口边界事件因时间戳不匹配而丢失
- 匹配依赖顺序/时间一致性，容易出错

**新方案**: `build_chain_seg_data.py` 直接从原始标注 JSON 构建：
- L2 caption 直接取自 `level2.events[i].instruction`
- L3 segments 通过 `parent_event_id` 直接关联，不依赖时间戳匹配
- 跨窗口边界事件正确裁剪，不会丢失

## 消融变体

| Variant | L2 任务 | L3 任务 | 输入 | problem_type |
|---------|---------|---------|------|-------------|
| **L2L3 (原始)** | 多 caption grounding | 自由分割 | 128s 视频 + 多条 caption | `temporal_seg_chain_L2L3` |
| **V1 (dual-seg)** | 自由分割 | 自由分割 | 128s 视频 | `temporal_seg_chain_dual_seg` |
| **V2 (ground-seg)** | 单 caption grounding | 自由分割 | 128s 视频 + 1 条 caption | `temporal_seg_chain_ground_seg` |

## 目录结构

```
chain_seg_ablation/
├── README.md                  # 本文件
├── build_chain_seg_data.py    # 从原始标注 JSON 构建 chain seg 数据
├── exp_chain_ablation.sh      # 单次实验入口 (L2L3/V1/V2)
└── run_chain_ablation.sh      # 批量运行所有变体
```

## 数据构建流程

```
原始标注 JSON (annotations/*.json)
  level2.events[i].instruction   → L2 caption
  level2.events[i].start/end     → L2 事件边界
  level3.grounding_results[i]    → L3 原子动作 (通过 parent_event_id 关联)
    ↓
build_chain_seg_data.py
  1. 生成 128s/64s 滑窗
  2. 裁剪 L2 事件到窗口边界
  3. 通过 event_id 关联 L3 结果
  4. 全部转换为窗口相对坐标 (0 ~ window_duration)
    ↓
chain_seg 训练数据 (各变体 train/val.jsonl)
  视频: 复用已有 clips/L2 下的 128s 窗口视频
```

## Reward 设计

共享文件: `verl/reward_function/youcook2_chain_seg_reward.py`

公式: `R = 0.4 × R_L2 + 0.6 × R_L3 × max(R_L2, 0.3)`

| | L2L3 (原始) | V1 (dual-seg) | V2 (ground-seg) |
|---|---|---|---|
| L2 评估 | F1-IoU (Hungarian 匹配) | F1-IoU (Hungarian 匹配) | temporal_iou (单段) |
| L3 评估 | 按 matched L2 配对 → F1-IoU | 按 matched L2 配对 → F1-IoU | clip 到 L2 bounds → F1-IoU |

## 运行

```bash
# 单个变体
VARIANT=L2L3 bash local_scripts/hier_seg_ablations/chain_seg_ablation/exp_chain_ablation.sh

VARIANT=V1 bash local_scripts/hier_seg_ablations/chain_seg_ablation/exp_chain_ablation.sh

VARIANT=V2 bash local_scripts/hier_seg_ablations/chain_seg_ablation/exp_chain_ablation.sh

# 批量 L2L3 → V1 → V2
bash local_scripts/hier_seg_ablations/chain_seg_ablation/run_chain_ablation.sh

# 仅准备数据
python3 local_scripts/hier_seg_ablations/chain_seg_ablation/build_chain_seg_data.py \
  --annotation-dir /path/to/annotations \
  --clip-dir /path/to/clips/L2 \
  --output-dir /tmp/chain_seg \
  --variants L2L3 V1 V2 \
  --complete-only

# 自定义路径
ANNOTATION_DIR=/custom/annotations CLIP_DIR=/custom/clips/L2 \
  VARIANT=V2 bash local_scripts/hier_seg_ablations/chain_seg_ablation/exp_chain_ablation.sh
```

## 数据量

- L2L3: ~窗口数 (128s 窗口, `--min-events 2`)，每窗口多事件
- V1: 同 L2L3 窗口数
- V2: 拆单事件后更多样本 (每 matched event 一条)

## 测试

```bash
python3 tests/test_chain_seg_reward.py
```
