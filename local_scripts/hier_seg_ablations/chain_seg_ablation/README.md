# Chain-Seg 实验 — V2 (ground-seg)

单 caption grounding (L2) + 单事件内原子动作分割 (L3)。

## 任务说明

- **输入**: 128s 视频 + 1 条 L2 caption
- **输出**: `<l2_events>[[start, end]]</l2_events>` + `<l3_events>[[[s1,e1], ...]]</l3_events>`
- **problem_type**: `temporal_seg_chain_ground_seg`

## Reward

```
R_L2 = temporal_iou(pred_l2, gt_l2)
R_L3 = F1-IoU(clipped_l3, gt_l3)  — 硬裁剪到 pred L2 边界
       如果 L3 有越界 pred L2 → R_L3 × 0.5
overall = 0.4 × R_L2 + 0.6 × R_L3
```

## 数据构建

```
原始标注 JSON → build_chain_seg_data.py → train/val.jsonl
  - 128s/64s 滑窗
  - 每个 matched event 拆为独立样本
  - 坐标全部转为窗口相对 (0 ~ duration)
```

## 运行

```bash
# 实验
bash local_scripts/hier_seg_ablations/chain_seg_ablation/exp_chain_ablation.sh

# 仅构建数据
python3 local_scripts/hier_seg_ablations/chain_seg_ablation/build_chain_seg_data.py \
  --annotation-dir /path/to/annotations \
  --clip-dir /path/to/clips/L2 \
  --output-dir /tmp/chain_seg \
  --complete-only
```

## 测试

```bash
python3 tests/test_chain_seg_reward.py
```
