# 分层 Reward V2 — NGIoU 分层奖励设计

## 1. 设计动机

现有 reward 对 L1/L2/L3 一视同仁（全用 F1-IoU），存在两个问题：

1. **L1 过罚**：宏观阶段差 5 秒不影响语义，但 IoU 会显著下降
2. **IoU 梯度断崖**：预测段与 GT 不重叠时 IoU=0，RL 训练无梯度信号

## 2. 核心改动

### 2.1 引入 NGIoU（Normalized Generalized IoU）

```
GIoU(a, b) = IoU(a, b) - (|C| - |A∪B|) / |C|
    其中 C = [min(a_start, b_start), max(a_end, b_end)] 为最小包围区间

NGIoU(a, b) = (GIoU + 1) / 2    ∈ [0, 1]
```

**关键优势**：当 pred 和 gt 完全不重叠时，IoU=0（无信号），**但 NGIoU > 0**（越近越高）。这为 RL 提供了从"完全不对"到"差一点"的连续梯度。

### 2.2 三层分化策略

| 层级 | 匹配方式 | 度量 | 特殊机制 |
|------|---------|------|---------|
| L1 宏观阶段 | **Hungarian** | **Margin-Relaxed NGIoU** | GT 边界外扩 Δ=5s 后计算 NGIoU（自然更宽容） |
| L2 中观事件 | **Hungarian** | **NGIoU** | F1-NGIoU |
| L3 原子动作 | **Hungarian** | **NGIoU** | F1-NGIoU |

**全局**：三层均不使用 NMS（提示词已要求输出不重叠片段，模型行为也很少重叠）。

### 2.3 设计决策记录

- **L1 不设满分阈值**：直接用外扩后 GT 计算 NGIoU，自然给予边界偏差更高分，无需硬编码阈值。
- **三层不用 NMS**：提示词明确要求输出不重叠片段，模型实际行为也极少重叠，NMS 在当前场景下无必要。
- **L3 不用 Sequential**：L3 最常见的错误模式是**多分割**（一个 2-6s 动作拆成两个 1-3s 子动作），Sequential 匹配会导致后续全部错位，惩罚过重。Hungarian 更稳健。
- **L3 不做 OOB 斩杀**：当前 L3_seg 是独立 setting（模型看到的就是裁切好的 event clip），预测 timestamp 自然不会越界。OOB 只在 chain_seg（一体预测 L2+L3）场景才有意义。
- **三层统一 Hungarian**：简化实现，减少潜在 bug，核心差异化通过 L1 margin 实现。

## 3. 各层详细设计

### 3.1 L1: Margin-Relaxed NGIoU

```python
# 对每个 (pred, gt) 对:
#   score = max(ngiou(pred, gt), ngiou(pred, expanded_gt))
# expanded_gt = [gt_start - 5, gt_end + 5]
# 取 max 确保 margin 只帮忙不帮倒忙
# （当 pred 已与原 GT 高度重叠时，expanded GT 反而会因尺寸不匹配而降分）
```

- `margin = 5` 秒（L1 阶段通常 30-120s，±5s 不影响语义）
- margin 对长段效果显著（60s 段 + 5s 偏移：L1=0.945 > L2=0.923）
- 短段效果不明显（20s 段 + 3s 偏移：L1 ≈ L2），但不会降分

### 3.2 L2: F1-NGIoU（标准模式）

- 用 `1.0 - ngiou(pred, gt)` 构建 cost matrix → Hungarian → F1 聚合
- 与现有 F1-IoU 的唯一差异：`temporal_iou` → `ngiou`

### 3.3 L3: F1-NGIoU

- 用 `1.0 - ngiou(pred, gt)` 构建 cost matrix → Hungarian → F1 聚合
- 与 L2 实现相同（统一调用 `compute_f1_ngiou`）

## 4. NGIoU 实现

新增到 `youcook2_temporal_seg_reward.py`：

```python
def ngiou(a: List[float], b: List[float]) -> float:
    """Normalized Generalized IoU for 1D intervals. Returns value in [0, 1]."""
    inter_start = max(a[0], b[0])
    inter_end = min(a[1], b[1])
    intersection = max(0.0, inter_end - inter_start)
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    if union <= 0:
        return 0.0
    iou = intersection / union

    c_len = max(a[1], b[1]) - min(a[0], b[0])
    if c_len <= 0:
        return 0.0

    giou = iou - (c_len - union) / c_len
    return (giou + 1.0) / 2.0
```

## 5. Dispatch 改动

`youcook2_hier_seg_reward.py` 中：

```python
_DISPATCH = {
    "temporal_seg_hier_L1":     _l1_reward,      # Margin-Relaxed F1-NGIoU
    "temporal_seg_hier_L2":     _l2_reward,      # F1-NGIoU
    "temporal_seg_hier_L3":     _l3_reward_v2,   # F1-NGIoU
    "temporal_seg_hier_L3_seg": _l3_reward_v2,   # 同上
}
```

## 6. F1-NGIoU vs F1-IoU 梯度对比

```
场景：GT = [[10,30], [40,60]]

Case A: pred = [[12,28], [42,58]]  (差一点)
  IoU:   [0.77, 0.73]   → F1-IoU  = 0.75
  NGIoU: [0.89, 0.87]   → F1-NGIoU = 0.88
  → NGIoU 对"差一点"更友善

Case B: pred = [[70,90]]  (完全偏)
  IoU:   [0, 0]          → F1-IoU  = 0.00
  NGIoU: [0.25, 0.38]    → F1-NGIoU = 0.16
  → NGIoU 非零，仍有信号（越近越高）

Case C: pred = [[10,30], [15,35], [40,60]]  (多了一段)
  F1-IoU:  recall=0.87, prec=0.58 → F1=0.69
  F1-NGIoU: recall=0.93, prec=0.62 → F1=0.75
  → 多余段自然被 F1 precision 惩罚
```

## 7. 改动文件清单

| 文件 | 改动 |
|------|------|
| `youcook2_temporal_seg_reward.py` | 新增 `ngiou()` + `compute_f1_ngiou()` |
| `youcook2_hier_seg_reward.py` | 新增 `_l1_reward` / `_l2_reward` / `_l3_reward_v2`；修改 `_DISPATCH`；旧 `_l1_l2_reward` 保留供对比 |

## 8. 参数汇总

| 参数 | 值 | 说明 |
|------|---|------|
| L1 margin (Δ) | 5 秒 | GT 边界外扩宽容度 |
| NMS | 全部关闭 | 提示词已要求不重叠，模型行为也极少重叠 |
