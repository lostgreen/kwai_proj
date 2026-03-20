# Proxy Data 构造工具集

基于 YouCook2 数据集构造的多种 Proxy 训练任务，用于视频时序理解和推理能力训练。所有任务输出统一为 **EasyR1 JSONL 格式**，可直接用于模型训练或混合使用。

---

## 一、项目背景与目标

目标是在 GRPO 强化学习框架（EasyR1）下，通过精心设计的 **Proxy 任务** 激励视频语言模型习得以下时序推理能力：

- **时序方向感知**：判断视频正放/倒放
- **事件边界定位**：在视频中找出事件的起止时间
- **层次化时序分割**：从粗到细（阶段→活动→动作）逐层理解视频结构
- **事件逻辑推理**：根据视频上下文选出下一步/替换/排序步骤

所有 Proxy 任务均以 **YouCook2** 烹饪视频数据集为基础，利用其精细的逐步骤标注。

---

## 二、Proxy 任务设计总览

| Proxy 类型 | 目录 | 任务描述 | problem_type | 状态 |
|-----------|------|---------|-------------|------|
| **Temporal AoT** | `temporal_aot/` | 判断视频片段正放/倒放方向（V2T / T2V） | `aot_v2t` / `aot_t2v` | ✅ 可用 |
| **Hierarchical Seg** | `youcook2_seg_annotation/` | 三层分层视频分割（阶段 / 活动 / 动作） | `temporal_seg_hier_L1/L2/L3` | ✅ 可用 |
| **Event Logic** | `event_logic/` | 基于文本选项的事件序列推理 | `add` / `replace` / `sort` | ⚠️ 需重构 |

---

## 三、各 Proxy 任务详细设计

### 3.1 Temporal Arrow of Time（时序方向判断）

**设计思路**：将真实视频片段翻转生成倒放版本，用 VLM 分别生成正放/倒放 caption。模型需要区分两者，从而学习时序方向感知能力。分为两种形式：

#### V2T（视频→文本选择）

| 项目 | 说明 |
|-----|------|
| **输入** | 一段视频片段（正放或倒放，随机选择一种），两条 caption 选项（A=正放描述，B=倒放描述） |
| **Prompt 结构** | `Watch the video... Which caption best matches the temporal direction? Options: A. {正放 caption} B. {倒放 caption}` |
| **输出格式** | `<think>...</think><answer>A</answer>` 或 `<answer>B</answer>` |
| **标准答案** | 固定为 `"A"`（正放视频对应选项 A） |
| **Videos 字段** | 正放视频路径 |

#### T2V（文本→视频选择）

| 项目 | 说明 |
|-----|------|
| **输入** | 拼接视频（正放段 + 黑屏间隔 + 倒放段），一条 caption（正放描述） |
| **Prompt 结构** | `The input video contains two segments separated by a black screen. Which segment best matches the caption "{正放 caption}"? Options: A. The first segment B. The second segment` |
| **输出格式** | `<think>...</think><answer>A</answer>` |
| **标准答案** | 固定为 `"A"`（第一段为正放） |
| **Videos 字段** | 合成的拼接视频路径 |

**数据生成流程**：
```
build_event_aot_data.py   → aot_event_manifest.jsonl（clip 路径 + 倒放视频生成）
annotate_event_captions.py → caption_pairs.jsonl（VLM 对正放/倒放各生成 caption）
build_aot_mcq.py          → V2T / T2V 训练 JSONL（过滤低置信度、相似度太高的 pair）
rebalance_aot_answers.py  → 对答案选项做重平衡（防止模型偏向某一选项）
```

---

### 3.2 Hierarchical Segmentation（三层分层时序分割）

**设计思路**：覆盖粗到细三个粒度，让模型从整体视频理解到精细动作定位，形成层次化时序感知能力。

#### L1 — 宏观阶段分割（Macro Phase Segmentation）

| 项目 | 说明 |
|-----|------|
| **粒度** | 阶段级（整段完整视频） |
| **训练样本数** | 100 条 |
| **输入视频** | 从 1fps 帧目录按 warped_mapping 均匀抽取 ≤256 帧合成的 mp4（1fps，M 秒） |
| **Prompt** | 给出帧编号 1..M（warped 帧号空间），要求将视频分割为 3–5 个高层语义阶段（如备料/烹炒/装盘） |
| **输出格式** | `<events>[[3, 80], [95, 150], [160, 220]]</events>`（warped 帧号区间） |
| **标准答案** | VLM 标注的 `macro_phases[]`（warped 帧号边界） |
| **关键设计** | warped_mapping 将真实帧号压缩到 ≤256 帧索引空间，确保帧坐标与模型所见一致 |

#### L2 — 滑窗事件检测（Sliding-Window Event Detection）

| 项目 | 说明 |
|-----|------|
| **粒度** | 活动级（128s 窗口） |
| **训练样本数** | 378 条 |
| **输入视频** | 从原始视频 ffmpeg 截取的 128s 子片段（时间归一化到 0s 起始） |
| **Prompt** | 说明片段为 0s to {duration}s，要求检测视频中所有完整的烹饪事件及其时间边界 |
| **输出格式** | `<events>[[5, 42], [55, 90]]</events>`（相对视频起点的秒数） |
| **标准答案** | YouCook2 标注 events（已减去 win_start 归一化） |
| **滑窗参数** | window=128s，stride=64s，min_events=2 |

#### L3 — 查询驱动原子动作定位（Query-conditioned Atomic Grounding）

| 项目 | 说明 |
|-----|------|
| **粒度** | 动作级（event clip 内） |
| **训练样本数** | 376 条（顺序/打乱各版本） |
| **输入视频** | 以 L2 event 边界为中心，±5s padding 截取子片段 |
| **Prompt** | 给出编号的原子动作 caption 列表（可打乱顺序），要求按列表顺序逐一输出时间区间 |
| **输出格式** | `<events>[[5, 8], [0, 4], [9, 13]]</events>`（按 query 列表顺序，每个动作对应一个 [start, end]） |
| **标准答案** | VLM 标注的 grounding_results（已减去 event_start 归一化） |
| **打乱设计** | 顺序版本（sequential）+ 打乱版本（shuffled），防止模型依赖序列先验 |

**数据生成流程**：
```
extract_frames.py    → 1fps 抽帧（frames/clip_key/）
annotate.py          → VLM 逐层标注（L1→L2→L3，annotations/clip_key.json）
build_dataset.py     → annotation → EasyR1 JSONL（youcook2_hier_L{1,2,3}_train.jsonl）
prepare_clips.py     → ffmpeg 截取子片段 + timestamps 归一化（*_clipped.jsonl）
sample_mixed_dataset.py → 三层按比例混合采样（youcook2_hier_mixed_train.jsonl）
```

---

### 3.3 Event Logic（事件逻辑推理）

**设计思路**：将 YouCook2 逐步骤标注转化为事件序列推理任务。选项以**文本描述**（而非视频片段）形式呈现，避免多视频拼接带来的复杂度，同时测试模型的语义和时序逻辑理解。

| 任务类型 | problem_type | 设计 |
|---------|-------------|------|
| **Add** | `add` | 给定连续 N 步视频上下文，选择下一步对应的 caption（从视频 + 文本选项中选一） |
| **Replace** | `replace` | 给定含缺失步的视频序列，选择正确填补的 caption（从文本选项中选一） |
| **Sort** | `sort` | 给定打乱顺序的视频片段，输出正确的时序排列数字序列 |

| 项目 | 说明 |
|-----|------|
| **输入** | 视频序列（event clips，1–N 个）+ 文本选项（A/B/C/D） |
| **输出格式（Add/Replace）** | `<think>...</think><answer>B</answer>` |
| **输出格式（Sort）** | `<think>...</think><answer>13245</answer>`（数字序列，代表正确时序） |
| **标准答案** | 选择题：`"A"/"B"/"C"/"D"`；排序题：数字序列字符串如 `"12345"` |

---

## 四、统一数据格式（EasyR1 JSONL）

所有 Proxy 任务输出均为 **EasyR1 JSONL** 格式，每行一条 JSON：

```json
{
  "messages": [{"role": "user", "content": "...（含 <video> 占位符）"}],
  "prompt": "...（同 messages[0].content）",
  "answer": "A",
  "videos": ["path/to/video.mp4"],
  "data_type": "video",
  "problem_type": "aot_v2t | aot_t2v | temporal_seg_hier_L1 | temporal_seg_hier_L2 | temporal_seg_hier_L3 | add | replace | sort",
  "metadata": {...}
}
```

| 字段 | 类型 | 说明 |
|-----|------|------|
| `messages` | list | 用户消息，content 中含 `<video>` 占位符 |
| `prompt` | str | 同 messages[0].content |
| `answer` | str | 标准答案（字母 / 时间区间 JSON / 数字序列） |
| `videos` | list[str] | 视频文件绝对路径（1 个或多个） |
| `data_type` | str | `"video"` |
| `problem_type` | str | 任务类型标识，驱动训练时的 reward 函数分发 |
| `metadata` | dict | 原始标注信息、视频来源、clip_key 等调试信息 |

---

## 五、奖励函数设计

奖励函数通过 `problem_type` 字段自动分发，统一接口为 `compute_score(reward_inputs) -> List[Dict]`，每条返回 `{"overall": float, "format": float, "accuracy": float}`。

### 5.1 选择题 Reward（add / replace / delete / aot_v2t / aot_t2v）

**策略**：严格格式门控 + 精确匹配。

```
有 <answer> 标签 + 答案正确 → overall=1.0, format=1.0, accuracy=1.0
有 <answer> 标签 + 答案错误 → overall=0.0, format=1.0, accuracy=0.0
无 <answer> 标签             → overall=0.0, format=0.0, accuracy=0.0
```

- 仅从 `<answer>...</answer>` 内提取字母，不做全文 fallback
- 多个标签时取最后一个

### 5.2 排序题 Reward（sort）— Temporal Jigsaw Displacement

**策略**：连续奖励，用序列位移距离衡量排序质量，不做二值化。

```
E_jigsaw = Σ |pos(k, P̂) - pos(k, P_gt)|      # 预测序列与 GT 的位置偏差之和
E_max    = Σ |i - (n-1-i)|  for i in 0..n-1   # 完全逆序时的最大位移（规范化分母）
R_jigsaw = max(0, 1 - E_jigsaw / E_max)        # [0, 1] 连续奖励
```

- 完全正确：R=1.0
- 完全逆序：R≈0.0
- 部分正确：R 介于 0 和 1 之间
- 需包含 `<answer>` 标签，缺失直接返回 0

### 5.3 时序分割 Reward（temporal_seg / temporal_seg_hier_L1 / L2）— F1-IoU

**策略**：NMS 去重 + 匈牙利最优匹配 + F1-IoU。

```
1. 1D NMS（IoU > 0.7 合并）去除重叠预测段
2. 构建 N_pred × N_gt IoU cost matrix
3. 匈牙利算法求最优一一匹配
4. recall    = Σ IoU_matched / N_gt
   precision = Σ IoU_matched / N_pred
   F1-IoU    = 2·R·P / (R+P)
```

- 输出格式必须包含 `<events>[[s,e],...]</events>`，缺失返回 0
- 反作弊：检测 `[数字-数字]` 畸形格式 / 重复 `<events>` 标签
- 不给格式保底分，避免 reward hacking，overall 完全对齐 F1-IoU

### 5.4 L3 Grounding Reward（temporal_seg_hier_L3）— Position-Aligned Mean tIoU

**策略**：按位置直接对齐，无需匈牙利匹配（因每个输出段对应唯一 query）。

```
mean_tIoU = Σ tIoU(pred[i], gt[i]) / max(N_pred, N_gt)
```

- 分母取 `max(N_pred, N_gt)`：少输出自动惩罚，多输出稀释精度
- 打乱 query 顺序版本同样按序对齐奖励，防止模型靠序列先验作答

### 5.5 统一 Dispatch 入口（mixed_proxy_reward.py / youcook2_hier_seg_reward.py）

| problem_type | Reward 函数 |
|---|---|
| `add` / `delete` / `replace` | 精确匹配选择题 |
| `aot_v2t` / `aot_t2v` | 精确匹配选择题 |
| `sort` | Jigsaw Displacement |
| `temporal_seg` | F1-IoU（NMS + 匈牙利） |
| `temporal_seg_hier_L1` / `L2` | F1-IoU（NMS + 匈牙利） |
| `temporal_seg_hier_L3` | Position-Aligned Mean tIoU |

---

## 六、目录结构

```
proxy_data/
├── temporal_aot/                        # 时序方向判断（Arrow of Time）
│   ├── build_event_aot_data.py          #   从 event clips 构造 AoT manifest + 生成倒放视频
│   ├── annotate_event_captions.py       #   对正放/倒放视频调用 VLM 生成 caption pairs
│   ├── build_aot_mcq.py                 #   构造 V2T / T2V 选择题训练数据
│   ├── rebalance_aot_answers.py         #   对答案选项做重平衡
│   ├── mix_aot_with_youcook2.py         #   将 AoT 数据与 YouCook2 base 数据混合
│   ├── prompts.py                       #   统一维护标注和 MCQ 用 prompt 模板
│   └── data/                            #   生成的数据文件
│       ├── aot_event_manifest.jsonl
│       ├── aot_event_manifest_all.jsonl
│       ├── aot_annotations/
│       ├── aot_invalid_clips.jsonl
│       ├── mixed_aot_train.jsonl
│       ├── mixed_aot_train.offline_filtered.jsonl
│       └── mixed_aot_val.jsonl
│
├── youcook2_seg_annotation/             # 三层分层视频分割标注（L1/L2/L3）
│   ├── extract_frames.py                #   ffmpeg 1fps 抽帧
│   ├── annotate.py                      #   VLM 标注流水线（逐层 L1→L2→L3）
│   ├── build_dataset.py                 #   标注 → EasyR1 训练 JSONL（含 warped 压缩）
│   ├── sample_mixed_dataset.py          #   对三层数据做比例采样混合
│   ├── prepare_clips.py                 #   ffmpeg 截取子片段 + timestamps 归一化
│   ├── prompts.py                       #   各层标注 prompt 模板
│   ├── run_build.sh                     #   一键构建脚本
│   ├── DESIGN.md                        #   架构设计文档（含已知问题和 Todo）
│   └── datasets/                        #   生成的训练数据
│       ├── youcook2_hier_L1_train_clipped.jsonl
│       ├── youcook2_hier_L2_train_clipped.jsonl
│       ├── youcook2_hier_L3_train_clipped.jsonl
│       ├── youcook2_hier_mixed_train.jsonl
│       └── youcook2_hier_mixed_val.jsonl
│
├── event_logic/                         # 文本选项事件逻辑推理
│   ├── build_text_option_proxy.py       #   构造 add / replace 文本选项任务
│   ├── redesign_prompts.py              #   重构 prompt 模板
│   ├── merge_datasets.py                #   合并 proxy + seg 数据
│   ├── convert_msswift_to_easyr1.py     #   MS-Swift 格式转换为 EasyR1 格式
│   ├── filter_bad_videos.py             #   过滤损坏视频
│   ├── validate_videos.py               #   视频有效性校验
│   └── data/                            #   生成的数据文件
│       ├── proxy_train_text_options.jsonl
│       ├── proxy_val_text_options.jsonl
│       ├── mixed_train_cot_clean.jsonl
│       └── ...
│
├── youcookii_annotations_trainval.json  # YouCook2 原始标注（trainval 全量）
├── youcook2_train_easyr1.jsonl          # YouCook2 时序分割训练数据（EasyR1 格式）
├── youcook2_val_small.jsonl             # YouCook2 验证集（小规模）
└── bad_videos.txt                       # 已知损坏视频列表
```

---

## 七、Reward 函数文件位置

| 文件 | 覆盖任务 |
|-----|---------|
| `verl/reward_function/mixed_proxy_reward.py` | add / delete / replace / sort / aot_v2t / aot_t2v / temporal_seg |
| `verl/reward_function/youcook2_hier_seg_reward.py` | temporal_seg_hier_L1 / L2 / L3 |
| `verl/reward_function/youcook2_temporal_seg_reward.py` | temporal_seg（底层 F1-IoU 工具函数，被上两者复用） |

---

## 八、数据规模统计

| 任务 | 数据集文件 | 样本数（约） |
|-----|-----------|------------|
| AoT V2T | `temporal_aot/data/mixed_aot_train.jsonl` | — |
| AoT T2V | `temporal_aot/data/mixed_aot_train.jsonl` | — |
| Hier Seg L1 | `youcook2_seg_annotation/datasets/youcook2_hier_L1_train_clipped.jsonl` | 100 |
| Hier Seg L2 | `youcook2_seg_annotation/datasets/youcook2_hier_L2_train_clipped.jsonl` | 378 |
| Hier Seg L3 | `youcook2_seg_annotation/datasets/youcook2_hier_L3_train_clipped.jsonl` | 376 |
| Hier Seg Mixed | `youcook2_seg_annotation/datasets/youcook2_hier_mixed_train.jsonl` | ~854 |
| Event Logic | `event_logic/data/proxy_train_text_options.jsonl` | — |
| YouCook2 Base | `youcook2_train_easyr1.jsonl` | — |

