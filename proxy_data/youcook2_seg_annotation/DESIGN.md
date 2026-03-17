# YouCook2 分层标注数据集设计文档

> 当前状态描述 + 已知问题 + 待实现工作
> 基于 `datasets/` 目录下 100+378+376 条实际数据整理

---

## 零、周报汇报摘要

基于 YouCook2 烹饪视频数据集，构建了一套**三层分层时序理解训练数据**，覆盖粗到细三个粒度的视频分割与定位任务，共产出训练样本 **100 + 378 + 376 条**（L1 + L2 + L3）。

### 三层任务总览

| 层级 | 任务名称 | 粒度 | 训练样本数 |
|---|---|---|---|
| L1 | Macro Phase Segmentation | 阶段级（整段视频） | 100 |
| L2 | Sliding-Window Event Detection | 活动级（128s 窗口） | 378 |
| L3 | Query-conditioned Atomic Grounding | 动作级（event clip） | 376（×2 顺/乱序） |

---

### L1 — 宏观阶段分割（Macro Phase Segmentation）

**任务**：给定一段完整烹饪视频，将其分割为 3–5 个高层语义阶段（如备料、烹炒、装盘）。

**输入格式**
- 视频：完整原始视频，均匀采样至 ≤256 帧（warped 压缩）
- Prompt：给出帧编号 1..N（warped 帧号），要求输出帧号区间

**输出格式**
```
<events>[[3, 80], [95, 150], [160, 220]]</events>
```

**奖励设计**
- 基于 **warped 帧号的 Temporal IoU（tIoU）**：将预测 phases 与 GT phases 做贪心最大匹配，计算匹配对的 tIoU 均值
- Format reward：`<events>` 格式合法 + pair 数在合理范围内

---

### L2 — 滑窗事件检测（Sliding-Window Event Detection）

**任务**：给定一个 128s 的烹饪视频片段，检测出其中所有完整的烹饪事件（cooking events）并给出时间边界。

**输入格式**
- 视频：从原始视频截取的 128s 子片段（ffmpeg 提取，≤128 帧），时间从 0s 开始
- Prompt：说明片段时长（0s to {duration}s），要求检测 events

**输出格式**
```
<events>[[5, 42], [55, 90]]</events>
```

**奖励设计**
- 基于 **Temporal IoU 匹配**：匈牙利算法将预测 events 与 GT events 做最优匹配，计算匹配对的平均 tIoU
- 可设阈值（如 tIoU ≥ 0.5）计算 **Recall@IoU0.5** 作为奖励信号
- Format reward：`<events>` 格式合法 + pair 数在合理范围内

---

### L3 — 查询驱动原子动作定位（Query-conditioned Atomic Grounding）

**任务**：给定 event 片段视频（event 边界 ± padding，≤128s）和一组原子动作 caption（顺序或打乱），要求模型按给定顺序逐一输出每个动作的时间区间。

**输入格式**
- 视频：以 L2 event 边界为中心，前后各加 5s padding，截取子片段（0-based）
- Prompt：给出编号动作列表（可打乱顺序），要求按列表顺序输出时间
  ```
  Actions to locate:
  1. "Slice the crushed garlic into smaller pieces"
  2. "Crush garlic cloves with the flat side of a knife"
  3. "Mince the sliced garlic using a rocking chop motion"
  ```

**输出格式**
```
<events>[[5, 8], [0, 4], [9, 13]]</events>
```
（按照 query 列表顺序，每个 action 对应一个 `[start_time, end_time]`）

**奖励设计**
- 因列表位置与 GT 对齐，无需匹配：直接按序计算每个 grounding 的 **tIoU**，取全部 query 的均值作为 episode reward
- 打乱顺序版本：奖励与顺序无关，仅看视觉定位精度（防止模型依靠序列先验作弊）
- Format reward：`<events>` 格式合法 + pair 数量与 query 数一致

---

### 数据生成流程（已完成）

```
annotate.py (VLM 自动标注 L1→L2→L3)
    │
    ▼
build_dataset.py → youcook2_hier_L{1,2,3}_train.jsonl
    │               (L1: warped 压缩 ≤256 帧; L3: 0-based 时间归一化)
    ▼
prepare_clips.py → youcook2_hier_L{2,3}_train_clipped.jsonl
                   (ffmpeg 截取子片段，更新 videos 字段)
```

所有产出数据均符合 **EasyR1 JSONL 格式**，可直接用于训练。

---

## 一、整体流水线

```
原始视频 (YouCook2_mp4)
    │
    ▼ extract_frames.py  (ffmpeg 1fps → frame images)
帧图像目录 (frames/clip_key/)
    │
    ▼ annotate.py  (VLM API, 逐层标注)
标注 JSON (annotations/clip_key.json)
    │
    ▼ build_dataset.py  (annotation → EasyR1 JSONL)
训练数据 (datasets/youcook2_hier_L*.jsonl)
```

**数据来源**：原始完整 YouTube 视频（非 windowed clip），`source_mode: "full_video"`
例：`/m2v_intern/.../YouCook2_mp4/training/205/--bv0V6ZjWI.mp4`（332s）

**规模**：100 个 clip，完整标注 L1+L2+L3

---

## 二、各 Level 当前设计

### Level 1 — Macro Phase Segmentation（阶段级）

| 字段 | 值 |
|---|---|
| `problem_type` | `temporal_seg_hier_L1` |
| `videos` | 完整原始视频（全长，63s–706s） |
| `prompt` | 给模型看 N 个编号帧，要求在 warped 帧号空间做 phase 分割 |
| `answer` | `macro_phases[]`，以 start_frame/end_frame（warped 帧号）为边界 |
| 时间坐标系 | **Warped 帧号**（1–N，非真实秒数） |

**当前 warped_mapping 状态**：目前是 1:1 identity（warped_idx = real_sec），即没有做实际的时间压缩。
`n_warped_frames` = clip 时长（秒），直接等于帧数。

**⚠️ 已知问题：超帧限制严重**
- 100 条记录中 **58 条（58%）的 n_warped_frames > 256**
- clip 时长分布：63s – 706s，均值 307s
- EasyR1 推理时最多输入 256 帧。若 clip 为 332s @ 1fps，模型实际只看到 256 帧（EasyR1 内部均匀降采样），但 prompt 中写着"332 帧编号 1–332"→ **帧号坐标系与模型实际看到的帧不一致**
- **必须在 build 阶段做 warped 压缩**：从 N 帧中均匀抽取 min(N, 256) 帧，重建 warped_mapping，prompt 改为对应的帧数

---

### Level 2 — Event Detection with Sliding Window（活动级）

| 字段 | 值 |
|---|---|
| `problem_type` | `temporal_seg_hier_L2` |
| `videos` | **完整原始视频**（全长，63s–706s）⚠️ |
| `prompt` | "viewing frames from a cooking phase ({win_start}s to {win_end}s, duration {duration}s)" |
| `answer` | `events[]`，start_time/end_time 为**绝对秒数**（相对完整视频起点） |
| 时间坐标系 | **绝对秒数**（相对完整视频） |

**滑窗参数**：window_size=128s，stride=64s，min_events=2
**窗口分布**：63s–128s，均值约 4.6 events/window，共 378 条

**⚠️ 已知问题：视频未截取**
- `videos` 指向完整原始视频（可达 706s）
- 但 prompt 说模型在看"0s 到 128s"这段
- EasyR1 加载完整视频后会均匀采样到 256 帧，但这 256 帧覆盖整个 706s，不是训练目标窗口 [win_start, win_end]
- **必须截取视频**：从原始视频中提取 [win_start, win_end] 子片段，保存为新 mp4，更新 `videos` 字段

**时间坐标系讨论**：
- 截取后视频从 0s 开始，但 answer 中 events 的 start_time/end_time 是绝对时间（如 start_time=9, 而窗口从 0 开始）
- 若截取的是 [0, 128] 窗口：坐标系一致（绝对时间 = 相对时间）✓
- 若截取的是 [64, 128] 的窗口：截取后视频从 0s 开始，但 answer 中可能有 start_time=70（绝对），对应视频内第 6s → **需要减去 win_start 做归一化**
- prompt 中的 "{win_start}s to {win_end}s" 也需要改为 "0s to {duration}s"

---

### Level 3 — Atomic Temporal Grounding（动作级）

| 字段 | 值 |
|---|---|
| `problem_type` | `temporal_seg_hier_L3` |
| `videos` | **完整原始视频**（全长）⚠️ |
| `prompt` | "viewing frames from a cooking event clip ({event_start}s to {event_end}s)"，给出 event 的 instruction |
| `answer` | `grounding_results[]`，start_time/end_time 为**绝对秒数** |
| 时间坐标系 | **绝对秒数**（相对完整视频） |

**event 分布**：时长 6s–98s，均值 26s，均值 4.3 actions/event，共 376 条

**⚠️ 已知问题：视频未截取**
- 同 L2：`videos` 指向完整原始视频，但模型应只看 [event_start, event_end] 片段
- **必须截取视频**：提取 [event_start, event_end]，更新 `videos`，timestamps 归一化（减去 event_start）

**⚠️ 任务格式待重新设计**（见下节）

---

## 三、L3 任务格式设计意图（待实现）

当前 build 产出的 L3 是"free-form grounding"：给模型看一段 event clip，让模型自己找出所有原子动作。

**目标设计**：把标注好的 grounding_results 作为查询序列（打乱或按顺序），要求模型在给定视频片段中依次完成 temporal grounding：

```
输入：
  视频：[event clip，时长 14s]
  查询序列（可打乱顺序）：
    1. "Slice the crushed garlic into smaller pieces"
    2. "Crush garlic cloves with the flat side of a knife"
    3. "Mince the sliced garlic using a rocking chop motion"

输出：
  对查询序列中每个 action，给出 start_time / end_time
```

这样的设计：
- 将 free-form generation 转换成 query-conditioned grounding（更可评估）
- 打乱顺序版本：测试模型是否真正理解视觉内容，而非依赖序列先验
- 顺序版本：可作为更简单的对齐任务

**需要改动**：
- `prompts.py`：新增 `get_level3_grounding_prompt(event_start, event_end, queries: list[str])`
- `build_dataset.py`：L3 build 时随机打乱或保持顺序，生成 queries + 对应 answer timestamps

---

## 四、待办事项汇总

### 优先级 P0（不做无法训练）
- [x] **L2/L3 视频截取**：新建 `prepare_clips.py`，读取 jsonl，ffmpeg 截取子片段，timestamps 归一化（减去 win/event start），更新 `videos` 字段，写出新 jsonl（`*_clipped.jsonl`）
- [x] **L1 warped 压缩**：`build_dataset.py` 新增 `--max-frames 256`，`_subsample_warped_mapping()` 均匀抽帧，phase 边界线性重映射到新帧号空间

### 优先级 P1（影响训练质量）
- [x] **L3 任务格式重设计**：query-conditioned grounding，`--l3-order sequential/shuffled/both`，`--l3-padding` 控制 event 上下文窗口

### 优先级 P2
- [x] `run_build.sh` 末尾调用 `prepare_clips.py` 自动处理

---

## 五、各 Level 数据统计（当前）

| Level | 记录数 | 视频时长范围 | 核心时间窗口 | 帧数约束 |
|---|---|---|---|---|
| L1 | 100 | 63–706s | 全视频 | **58% 超 256 帧** |
| L2 | 378 | 原始 63–706s | 截取窗口 63–128s | 窗口内 ≤128 帧 ✓ |
| L3 | 376 | 原始 63–706s | 截取事件 6–98s | 事件内 ≤98 帧 ✓ |
