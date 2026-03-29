# Ablation Comparison — 多 Setting 分割对比可视化

> 在同一时间轴上并排比较 PA1/PA2/R1/R2 的分割时间戳，直观感受 setting 差异。

## 快速开始

```bash
# 1. 对比训练数据（不同 prompt → 不同数据格式）
bash ablation_comparison/run.sh --data

# 2. 对比 rollout 输出（不同模型 → 不同预测）
bash ablation_comparison/run.sh --rollout

# 3. 自定义
python3 ablation_comparison/server.py \
    --setting PA1:/path/to/pa1/data \
    --setting PA2:/path/to/pa2/data \
    --port 8790
```

打开 `http://localhost:8790/` 即可查看。

## 功能

- **时间轴对比**：GT (红色) + 各 setting 预测段，同一时间轴上并排显示
- **帧条同步**：1fps 视频帧 + 时间轴同步滚动
- **Prompt 对比**：不同 setting 的 prompt 并排展示差异
- **Response 对比**：模型输出文本 + reward 值
- **分割详情**：精确到秒的时间戳表格
- **筛选**：按 Level (L2/L3) 过滤、关键词搜索

## 数据格式

支持标准 JSONL 格式：

```json
{
  "video_paths": ["/path/to/video.mp4"],
  "prompt": "...",
  "response": "<events>[[10, 35], [40, 72]]</events>",
  "ground_truth": "<events>[[8, 38], [42, 70]]</events>",
  "reward": 0.85,
  "step": 100,
  "metadata": {"level": "L2", "duration": 128}
}
```

兼容 rollout_visualization 的 `step_*.jsonl` 格式和训练数据 `train.jsonl` 格式。

## 配色

| Setting | 颜色 |
|---------|------|
| GT | 🔴 红色 |
| 第1个 setting | 🔵 蓝色 |
| 第2个 setting | 🟢 绿色 |
| 第3个 setting | 🟠 橙色 |
| 第4个 setting | 🩷 粉色 |
