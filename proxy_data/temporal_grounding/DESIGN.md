# Temporal Grounding Proxy Task — 设计文档

> 将 Time-R1 的 TimeRFT 训练数据（2.5K 条 temporal video grounding）适配到 VideoProxy 的 256 frames / 2fps 训练方案中

---

## 一、背景

### 1.1 Time-R1 数据概况

**TimeRFT 训练集（train_2k5.json）**：2500 条 temporal grounding 标注，每条为一个完整视频中的一个事件时间段查询。

**原始数据字段**：
```json
{
  "video": "./dataset/timer1/videos/timerft_data/QtWT8rxUuNk.mp4",
  "duration": 93.29,
  "timestamp": [1.0, 4.0],
  "sentence": "A purple hanger demonstrates how to hang a sweater.",
  "qid": "my|cosmo|QtWT8rxUuNk|...",
  "video_start": null,      // 可选：原始视频裁切起点
  "video_end": null,         // 可选：原始视频裁切终点
  "difficulty": 50.0,
  "pred": [0.0, 3.0]        // 基线模型预测（用于筛选难度）
}
```

**数据来源混合**：Cosmo, YT-Temporal, DiDeMo, InternVid-VTime, TimeIT/VTG-IT, TimePro/HTStep, LongVid 等多个视频数据集。

**视频时长分布**：从 ~10s 到 ~600s+ 不等，大部分在 30s–200s 范围。

### 1.2 服务器数据路径

```
/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset/
├── README.md
├── timerft_data/           # TimeRFT 训练视频（2.5K 条对应的 mp4）
├── TimeRFT_data.zip
├── tvgbench_data/          # TVGBench 评估视频
└── TVGBench.zip
```

### 1.3 我们的训练方案约束

| 参数 | 值 | 说明 |
|------|------|------|
| **max_frames** | 256 | 每个视频最多采样 256 帧 |
| **video_fps** | 2.0 | 采样帧率 2fps |
| **最大视频时长覆盖** | 128s | 256 frames / 2fps = 128 秒 |
| **输出格式** | `<events>[[s, e]]</events>` | 统一 events 标签 |
| **奖励框架** | EasyR1 GRPO | batch reward，按 problem_type 分发 |

---

## 二、核心适配问题分析

### 2.1 视频时长 vs 帧数预算

**问题**：256 frames @ 2fps 只能覆盖 128 秒。但 TimeRFT 数据中存在超过 128 秒的视频（如 570s）。

**实际统计（train_2k5, 2500 条）**：

| 时长区间 | 数量 | 占比 | 备注 |
|----------|------|------|------|
| ≤ 30s | 239 | 9.6% | |
| 30–60s | 454 | 18.2% | |
| 60–128s | 913 | 36.5% | |
| 128–256s | 540 | 21.6% | ⚠️ 超 128s |
| 256–600s | 269 | 10.8% | ⚠️ 超 128s |
| > 600s | 85 | 3.4% | ⚠️ 超 128s |

- 平均时长 147.0s，最短 8.9s，最长 1154.4s
- 事件长度：平均 32.0s，最短 0.6s，最长 14100.0s（异常值）
- **~35.8% 的视频超过 128s**，需要框架自动降采样

**策略决策**：

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **A. 直接送入（推荐）** | 所有视频直接送入 process_video，由框架自动按 2fps + max_frames=256 采样 | 零额外处理；短视频帧数少、长视频自动降采样 | 长视频 > 128s 时，模型看到的帧间距 > 0.5s，时间分辨率降低 |
| **B. 裁切子片段 + 时间戳偏移** | 对长视频 > 128s，以 GT 事件为中心裁切 128s 窗口，时间戳归零 | 保证 2fps 精度 | 需预处理裁切；可能丢失上下文；GT 在边缘可能被截断 |
| **C. 过滤仅保留 ≤128s** | 只保留 duration ≤ 128s 的样本 | 最简单；保证时间精度 | 丢失大量训练数据 |

**推荐方案 A**：直接送入框架，不做视频裁切。理由：
1. Time-R1 原始方案中 FPS_MAX_FRAMES=768，对 570s 视频也只采到 186 帧（远低于 768）。我们 256 帧对多数视频也够用。
2. 长视频中模型学到的是"在更稀疏的帧序列中做粗粒度定位"，这与我们 L1/L2 任务互补。
3. 框架 `process_video` 已经能正确处理任意长度视频（自动调整 fps）。
4. 模型接收的帧带有 fps 信息，可以自动推算秒数。

### 2.2 时间戳精度 — 是否需要整数化？

**Time-R1 原始精度**：标注精确到 0.01 秒（如 `[13.4, 28.1]`），模型输出也要求两位小数。

**我们的框架现状**：
- L1 任务：整数帧号（warped 帧索引）
- L2/L3 任务：整数秒 `[[5, 42], [55, 90]]`

**决策：保留浮点精度，不做整数化。理由**：
1. Temporal grounding 的精度本就需要亚秒级——整数化会引入 ±0.5s 的系统误差。短事件（如 `[1.0, 4.0]`，仅 3s）整数化后 IoU 显著下降。
2. 本任务 prompt 中会明确说明浮点秒数格式，与整数格式的 L1/L2 任务互不影响。
3. 奖励函数独立，可以支持浮点比较。

### 2.3 输出格式统一

**Time-R1 原始格式**：
```
<think>...reasoning...</think>
<answer>12.54 to 17.83</answer>
```

**我们的框架格式**：
```
<think>...reasoning...</think>
<answer>
<events>[[12.54, 17.83]]</events>
</answer>
```

**决策：采用 `<events>` 统一格式**，即：

```
<events>[[start_time, end_time]]</events>
```

理由：
1. 与框架已有的 temporal_seg 任务格式统一，复用解析/奖励代码。
2. 单事件定位的 `<events>` 只有一个 pair，是多事件检测的特例，概念自然兼容。
3. 避免模型在不同任务间切换格式，降低格式混淆。

### 2.4 Prompt 设计

改造 Time-R1 的 prompt 模板，适配 `<events>` 输出格式：

```
Watch the following video carefully:
<video>

This video is {duration:.1f} seconds long.

Your task: Locate the time period of the event "{sentence}" in this video.

Output format (strictly follow this):
<events>
[start_time, end_time]
</events>

Where start_time and end_time are in seconds (precise to one decimal place, e.g., [12.5, 17.8]).
```

**关键设计点**：
- 明确告知视频总时长，帮助模型建立时间参照系
- 要求一位小数精度（不是两位），因为 2fps 下时间分辨率为 0.5s，两位小数无意义
- 单个 `[start, end]` pair

---

## 三、数据转换方案

### 3.1 输入：Time-R1 train_2k5.json

**服务器路径**：`/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset/timerft_data/`

### 3.2 输出：EasyR1 JSONL

文件：`proxy_data/temporal_grounding/data/timerft_train_easyr1.jsonl`

每行格式：
```json
{
  "messages": [
    {"role": "user", "content": "Watch the following video carefully:\n<video>\n\nThis video is 93.3 seconds long.\n\nYour task: Locate the time period of the event \"A purple hanger demonstrates how to hang a sweater.\" in this video.\n\nOutput format (strictly follow this):\n<events>\n[start_time, end_time]\n</events>\n\nWhere start_time and end_time are in seconds (precise to one decimal place, e.g., [12.5, 17.8])."}
  ],
  "prompt": "<same as messages[0].content>",
  "answer": "<events>\n[1.0, 4.0]\n</events>",
  "videos": ["/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset/timerft_data/QtWT8rxUuNk.mp4"],
  "data_type": "video",
  "problem_type": "temporal_grounding",
  "metadata": {
    "video_id": "QtWT8rxUuNk",
    "duration": 93.29,
    "timestamp": [1.0, 4.0],
    "sentence": "A purple hanger demonstrates how to hang a sweater.",
    "source": "cosmo",
    "difficulty": 50.0,
    "qid": "my|cosmo|QtWT8rxUuNk|..."
  }
}
```

### 3.3 视频路径映射

原始路径 `./dataset/timer1/videos/timerft_data/XXX.mp4` → 服务器路径：
```
/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset/timerft_data/XXX.mp4
```

提取规则：
```python
video_filename = os.path.basename(item["video"])  # e.g. "QtWT8rxUuNk.mp4"
server_path = f"/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset/timerft_data/{video_filename}"
```

### 3.4 `video_start` / `video_end` 处理

Time-R1 部分样本有 `video_start` 和 `video_end` 字段（非 null），表示原始视频需要裁切。

**处理方式**：
- 如果 `video_start` 和 `video_end` 都不为 null：
  - 有效视频区间为 `[video_start, video_end]`，有效时长 = `video_end - video_start`
  - **方案 1（推荐）**：在 prompt 中使用有效时长（即 `duration` 字段已标注的值），GT timestamp 相对于有效区间起点
  - 视频路径不变，但在 metadata 中记录 `clip_start` / `clip_end`，由 `process_video` 在加载时裁切
  - **注意**：需确认我们的 `process_video` 是否支持 `video_start`/`video_end` 参数。如不支持，需用 ffmpeg 预裁切视频。
- 如果两者都为 null：直接使用原始视频和 duration

### 3.5 转换脚本

```
proxy_data/temporal_grounding/
├── DESIGN.md                     # 本文档
├── build_dataset.py              # Time-R1 JSON → EasyR1 JSONL 转换
├── data/
│   ├── timerft_train_easyr1.jsonl     # 训练集（~2500 条）
│   └── tvgbench_val_easyr1.jsonl      # 评估集（TVGBench 800 条）
└── README.md                     # 使用说明
```

---

## 四、奖励函数设计

### 4.1 新增 problem_type: `temporal_grounding`

奖励函数文件：`verl/reward_function/temporal_grounding_reward.py`

### 4.2 奖励计算

采用 **IoU 奖励 + 格式奖励** 的组合，与 Time-R1 的 `iou_v2` 对齐但适配 `<events>` 格式：

```python
def compute_temporal_grounding_reward(response: str, ground_truth: str, metadata: dict) -> dict:
    """
    解析 <events>[[start, end]]</events> 格式，计算 IoU 奖励
    """
    # 1. 解析预测
    pred_segments = parse_events_tag(response)  # 复用已有解析逻辑
    if pred_segments is None or len(pred_segments) == 0:
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    # 2. 格式合法
    format_score = 1.0

    # 3. 取第一个 segment 作为预测
    pred_start, pred_end = pred_segments[0]

    # 4. 解析 GT
    gt_segments = parse_events_tag(ground_truth)
    gt_start, gt_end = gt_segments[0]

    # 5. IoU 计算
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    iou = intersection / union if union > 0 else 0.0

    # 6. 归一化距离惩罚（参考 Time-R1 iou_v2）
    duration = metadata.get("duration", 1.0)
    if duration > 0:
        dist_penalty = (
            (1 - abs(gt_start / duration - pred_start / duration)) *
            (1 - abs(gt_end / duration - pred_end / duration))
        )
        accuracy = iou * dist_penalty
    else:
        accuracy = iou

    return {
        "overall": accuracy,
        "format": format_score,
        "accuracy": accuracy
    }
```

### 4.3 集成到 mixed_proxy_reward.py

在奖励分发逻辑中新增：
```python
if problem_type == "temporal_grounding":
    return compute_temporal_grounding_reward(response, ground_truth, metadata)
```

---

## 五、TVGBench 评估集适配

TVGBench 包含 800 条评估样本（Charades + ActivityNet + DiDeMo 混合），可作为验证集。

**格式转换**：
```json
{
  "messages": [...],
  "prompt": "...",
  "answer": "<events>\n[13.4, 28.1]\n</events>",
  "videos": ["/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset/tvgbench_data/KWZSA.mp4"],
  "data_type": "video",
  "problem_type": "temporal_grounding",
  "metadata": {
    "duration": 30.12,
    "timestamp": [13.4, 28.1],
    "sentence": "a person was sitting at a table.",
    "dataset_name": "charades",
    "qsemtype": "Human Pose"
  }
}
```

---

## 六、实现清单

### Phase 1：数据转换（本次完成）

- [ ] `build_dataset.py`：将 `train_2k5.json` 转换为 `timerft_train_easyr1.jsonl`
  - 路径映射：`./dataset/timer1/videos/timerft_data/XXX.mp4` → 服务器绝对路径
  - Prompt 生成：插入 duration 和 sentence
  - Answer 生成：`<events>\n[start, end]\n</events>`
  - Metadata 保留：duration, difficulty, source, qid 等
  - 统计输出：视频时长分布、timestamp 范围分布
- [ ] `build_dataset.py`：将 `tvgbench.json` 转换为 `tvgbench_val_easyr1.jsonl`
  - 路径映射：`./dataset/timer1/videos/tvgbench_data/XXX.mp4` → 服务器绝对路径
  - Answer 解析：`"13.4-28.1"` → `<events>\n[13.4, 28.1]\n</events>`

### Phase 2：奖励函数

- [ ] 新增 `verl/reward_function/temporal_grounding_reward.py`
- [ ] 在 `mixed_proxy_reward.py` 中注册 `temporal_grounding` problem_type
- [ ] 单元测试：IoU 计算正确性、格式解析边界情况

### Phase 3：训练集成

- [ ] 在 local_scripts 中添加 temporal grounding 混合训练脚本
- [ ] 验证 `process_video` 对长视频（>128s）的帧采样行为
- [ ] 确认 `video_start/video_end` 在框架中的支持情况（如不支持，预裁切视频）
- [ ] 混合训练调试：temporal_grounding + 现有 proxy 任务的混合比例

### Phase 4：评估

- [ ] TVGBench 评估 pipeline
- [ ] 输出 mIoU / R@0.3 / R@0.5 / R@0.7 指标

---

## 七、关键决策摘要

| 问题 | 决策 | 理由 |
|------|------|------|
| 长视频是否裁切 | **不裁切**，直接送入框架 | 框架自动降采样；保留上下文 |
| 时间戳是否整数化 | **不整数化**，保留 1 位小数 | 亚秒精度对短事件定位至关重要 |
| 输出格式 | **统一 `<events>` 标签** | 复用解析/奖励代码，降低格式混淆 |
| 视频路径 | **服务器绝对路径** | 与现有数据一致 |
| 奖励函数 | **IoU × 归一化距离惩罚**（iou_v2） | 参考 Time-R1 最优配置 |
| Prompt 是否包含 CoT | **不强制 CoT**，由模型自行决定 | 与现有 proxy 任务风格一致 |

---

## 八、风险与 TODO

1. **video_start/video_end 兼容性**：需确认 `verl/utils/dataset.py` 的 `process_video` 是否支持时间裁切参数。若不支持，有两个备选：
   - 方案 a：用 ffmpeg 预裁切这些视频（~几百个），存入独立目录
   - 方案 b：扩展 `process_video` 支持 `start_time`/`end_time` 参数

2. **超长视频的时间分辨率下降**：当视频 > 128s 时，2fps+256 帧 = 只能覆盖 128s，框架会降低 fps。一个 570s 的视频大约采样到 0.45fps（256/570），帧间距 ~2.2s。对于精确到 1s 以内的事件定位，这可能不够。留待实验验证。

3. **数据量平衡**：TimeRFT 有 2500 条，现有 proxy 数据总共约 1500+ 条。混合训练时需关注采样比例，避免某类任务被淹没。建议初始比例 temporal_grounding:其他 = 1:1。
