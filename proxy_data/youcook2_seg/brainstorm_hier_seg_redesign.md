# Hierarchical Segmentation Task Redesign — Brainstorm

> 记录日期：2026-04-03

---

## 背景问题

当前三层分割标注流程（L1 phase / L2 1/ L3 action clip）过于严格，导致筛选下来的数据主要集中在教学类视频（YooCook2 cooking 类型）。

**核心设计目标**：引导模型在不同粒度上采集和处理信息、理解事件边界，最终支撑多事件复杂视频的关系推理能力。

---

## 现有任务的收益与局限

### 有效的部分

- **多尺度时序感知**：L1/L2/L3 分别要求模型在三个尺度上标出边界，迫使模型根据 prompt 粒度调整注意力范围
- **3x 监督密度**：同一视频产生三条样本，边界定位能力被强化
- **边界感知基础**：事件关系推理的前提条件

### 根本性局限

1. **层次之间割裂**：三层独立训练、独立 reward，模型学到的是"粗粒度 prompt → 少段输出；细粒度 prompt → 多段输出"，但并没有被要求理解 L3 的 segment 属于哪个 L2 事件（part-whole 关系）

2. **训练"在哪里"而不是"为什么"**：F1-IoU reward 只奖励边界位置准确性，模型可以靠镜头切换等视觉 shortcut 得高分，而无需理解事件内容

3. **数据单一**：三层完整标注的要求自然筛掉了非结构化视频

---

## 核心诊断

当前 pipeline 能力分工：

```
分割任务 → "边界在哪里" (位置感知)
AOT 任务  → "事件顺序是什么" (时序关系)

缺失的能力：
  ↓
  "哪些事件属于同一个更高层次的事件" (层次包含关系)
  "这些事件为什么在一起" (语义聚合)
```

---

## 改造方向

### 方向一：相对粒度替换绝对层次

不再要求视频有 L1/L2/L3 三个固定层次，改为按视频自身结构定义粗/中/细三档：

```
粗粒度：3~5 个 segment，每段 20%+ 的视频时长
中粒度：8~15 个 segment，每段 10%~20%
细粒度：15+ 个 segment，每段 <10%
```

任意视频只要能产生两种以上粒度划分就可入库，不要求三层都有。

### 方向二：弱监督信号扩展来源

| 信号 | 来源 | 对应粒度 |
|------|------|---------|
| YouTube chapters | 视频元数据，免费 | 粗粒度（L1 等价） |
| Shot boundary detection (PySceneDetect / TransNetV2) | 算法，任意视频 | 细粒度（L3 等价） |
| ASR topic shift (WhisperX + topic segmentation) | 任意有语音视频 | 中粒度（L2 等价） |

这三个信号可自动组合成两级或三级粗标注，适用于任意视频来源。

### 方向三：自适应层数标注（最小改动）

把"必须有三层"改为"至少有两层，最多有三层"：

```python
if duration < 120s:  → 1-2 levels
elif duration < 600s: → 2 levels
else:                → 2-3 levels (optional L3)
```

### 方向四（新任务类型）：Hierarchical Grouping Task ⭐

**核心思路**：增加一类直接训练"层次关系"的任务，填补现有 pipeline 中跨层次关系推理的空白。

---

## Hierarchical Grouping Task 详细设计

### 确定方案：自由 N + Bottom-up

**Bottom-up Grouping（细→粗归并，自由决定组数）**：

```
输入：
  视频片段 + 预给定的细粒度时间戳列表（GT 或 pseudo-GT）
  [0-10s, 10-25s, 25-35s, 35-50s, 50-65s, 65-80s, 80-95s, 95-120s]

Prompt 示例（不指定组数）：
  "The video has been pre-segmented into the following fine-grained clips.
   Group them into higher-level events based on the content.
   Output the time boundaries of each high-level event."

输出：粗粒度边界（数量由模型自己决定）
  [0-35s, 35-80s, 80-120s]
```

`problem_type` 命名建议：`hier_grouping_bu`

### Reward 设计

**直接复用 `_l1_l2_reward`（F1-IoU），无需修改**。

自由 N 不需要特殊处理：
- 模型分太多组 → precision 低（预测段没有 GT 对应）
- 模型分太少组 → recall 低（GT 段没有被预测覆盖）
- F1 自动平衡，自然惩罚错误的 N

```python
# 现有 reward 直接适用，ground-truth = 对应粗粒度标注
reward = _l1_l2_reward(pred_segments, gt_coarse_segments)
```

### 数据生成策略

**从现有层次数据**：
- L3 segments → 归组到 L2（已有 GT，直接可用）
- L2 segments → 归组到 L1（同上）

**从新的多元数据（自动生成）**：
- PySceneDetect shot boundaries → 细粒度输入
- YouTube chapters → 粗粒度 GT
- 任意 YouTube 有 chapters 的视频都可以用 → 大量多元数据

### 与现有 Pipeline 的关系

```
现有任务                 新增 Grouping 任务
──────────────          ──────────────────
L1 分割（粗边界定位）      Bottom-up Grouping（细→粗关系推理）
L2 分割（中边界定位）      Top-down Attribution（粗段内细段归属）
L3 分割（细边界定位）
AOT（事件顺序）
```

Grouping 任务填补的是**跨层次关系**这一空白，而不是替换现有任务。

---

## 已确认的设计选择

| 问题 | 决策 |
|------|------|
| 粗粒度组数 | **自由 N**（模型自己决定，F1-IoU 天然支持） |
| 任务变体优先 | **Bottom-up**（细→粗归并） |

## 待确认的设计选择

| 问题 | 选项 A | 选项 B |
|------|--------|--------|
| 与现有任务比例 | 补充（少量，验证效果） | 替换部分分割任务 |
| 数据来源优先 | 现有 YooCook2 层次数据 | YouTube chapters + shot detection |

---

---

## 方向五（新想法）：Few-shot 粒度校准

**核心思路**：不用文字描述"中等粒度"，而是通过 few-shot 例子让模型看到目标粒度的样子。

### 两种实现方式

**纯文本 Few-shot**：
```
Context examples:
  Event 1: [5-18s] — person picks up ingredients
  Event 2: [18-32s] — chops vegetables
  Event 3: [32-45s] — heats pan

Now segment [45-110s] into events of similar granularity.
```
- 优点：轻量，可扩展到任意视频（只需相邻段的时间戳）
- 局限：只能校准时长/数量，无法校准语义边界

**视频 Few-shot（完整 clip）**：
- 相邻 events 的实际视频片段作为示例
- 模型能学到"边界出现的语义/视觉模式"
- 代价高：多个 clip 进 input，token 成本可能翻倍（Qwen3-VL-4B）

**关键帧 + 文本（中间方案）**：
```
Context events (one keyframe each):
  [5-18s]: [image_frame_at_10s]  — picks up ingredients
  [18-32s]: [image_frame_at_25s] — chops vegetables
  [32-45s]: [image_frame_at_38s] — heats pan
```
- 语义信息有了，token 成本远低于完整视频片段
- 需要关键帧提取策略（中间帧 or 最具代表性帧）

### 待确认细节（稍后讨论）

- [ ] 优先使用哪种 few-shot 形式（纯文本 / 关键帧+文本 / 完整视频）
- [ ] few-shot 例子从哪里来（同视频相邻段 / 跨视频采样）
- [ ] 风险：如果 few-shot 来自同视频，模型可能 copy 边界风格而非真正理解内容
- [ ] 例子数量（1-3 个 reference events）

---

---

## 可扩展的数据源

### 当前使用（已过严格三层筛选）
- **TimeLens-100K**：Gemini-2.5-Pro 重标注的时序 grounding 数据，来自 cosmo_cap / didemo / hirest / internvid_vtime 等
- **ET-Instruct 164K**：14 个源数据集混合，覆盖 8 个领域的事件级理解任务
- **筛选后剩余**：~8K 视频，领域偏教学类

### 推荐扩展来源

| 数据集 | 规模 | 层次结构 | 领域 | 备注 |
|--------|------|---------|------|------|
| [Action100M](https://arxiv.org/html/2601.10592) | 1.2M 视频 / 147M 段 | 层次化（procedural step 多层） | 程序类（极宽） | 自动化标注 pipeline，适合借鉴 |
| [HT-Step](https://proceedings.neurips.cc/paper_files/paper/2023/file/9d58d85bfc041b4f901c62ba37a3f322-Paper-Datasets_and_Benchmarks.pdf) | 116K step labels / 20K videos | 步骤级（HowTo100M 子集） | 多样 how-to | 基于 wikiHow 对齐，步骤边界质量高 |
| [NewsNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_NewsNet_A_Novel_Dataset_for_Hierarchical_Temporal_Segmentation_CVPR_2023_paper.pdf) | 1,000 videos / 900h | 4 层（event/scene/story/topic） | 新闻 | 最完整的层次化时序标注，CVPR 2023 |
| [Assembly101](https://assembly-101.github.io/) | 4,321 videos | 粗+细（100K coarse / 1M fine） | 装配操作 | 跨域程序类，非烹饪 |
| [FineSports CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_FineSports_A_Multi-person_Hierarchical_Sports_Video_Dataset_for_Fine-grained_Action_CVPR_2024_paper.pdf) | 10K NBA videos | 2 层（action type / sub-action） | 体育 | 运动领域层次化标注 |
| ActivityNet Captions | 20K videos | 事件级密集 caption | 多领域活动 | 经典基准，时间戳+文本 |

**最推荐引入**：HT-Step（多样 how-to，层次清晰）+ NewsNet（非教学领域，4 层结构）+ Action100M 的标注 pipeline 方法论

---

## 标注流程优化

### 当前流程问题

```
现有流程（自顶向下 + 向上验证）：
  ① 严格筛选视频（高门槛，过滤大量数据）
  ② L1 标注（人工/VLM）
  ③ L2 标注（基于 L1，人工/VLM）
  ④ L3 标注（基于 L2，人工/VLM）
  ⑤ 向上 check（验证 L3 与 L2 的一致性）← 成本最高
```

### 简化方向

**方案 A：双轨自动化（推荐）**

去掉向上 check，改为两路自动信号 + VLM 填中间层：

```
① 粗粒度（L1）：YouTube chapters / ASR topic shift → 自动，免费
② 细粒度（L3）：PySceneDetect shot boundary → 自动，免费
③ 中粒度（L2）：VLM 只标注每个 L1 segment 内部 → 成本降低 N 倍
   （不需要看全视频，每次输入是一个 L1 片段）

优点：
  - 无需向上 check（L1 和 L3 本来就是自动的）
  - VLM 标注成本大幅下降
  - 可扩展到任意有 chapters/speech 的视频
```

**方案 B：只标注 2 层 + Grouping 任务补充**

- L1（coarse）：自动
- L2（event）：VLM 标注
- 取消 L3，改用 Grouping 任务（L2→L1 的归并）代替层次感知

**方案 C：Action100M pipeline 方法论（最彻底）**

参考 Action100M 的做法：
1. 先用 ASR/shot 把视频转成 "文本层次树"（Tree of Captions）
2. VLM/LLM 在纯文本树上做层次对齐
3. 最后才映射回时间戳

这样 VLM 不用处理原始视频帧，成本极低，质量靠 LLM 文本推理保证

---

## 确定方案：分割训练 L1/L2 + L3 伪标注做 Bottom-up Grouping ⭐

**核心决策**：

```
标注侧：只做 L1（粗边界）和 L2（事件边界）的真实标注
         ↓
L3 侧：在每个 L2 segment 内，用自动方式生成"伪 L3"子段
         ↓
训练任务：Bottom-up Grouping（给模型伪 L3 段列表，目标归并回 L2）
         ↓
Reward：F1-IoU 对比归并结果 vs L2 GT（直接复用现有 reward）
```

### 伪 L3 生成策略（推荐优先级）

1. **Shot boundary**（PySceneDetect）：子段语义最自然
2. **等时长切分**：简单可靠，作为 fallback（L2 内 shot 数 < 2 时）
3. 不推荐纯随机切割

### 优势

- **标注成本**：L1+L2 只需 2 层标注，L3 完全自动化
- **向上 check 消除**：L3 构造在 L2 以内，天然一致，无需验证
- **数据多元化**：只需"有 L1+L2 的视频"即可，可接入 HT-Step、Shot2Story 等
- **Reward 兼容**：F1-IoU 直接复用，无需改 reward function

### 与现有分割任务的关系

```
保留：
  L1 分割任务（粗粒度边界定位，正向监督）
  L2 分割任务（中粒度边界定位，正向监督）

新增：
  L2→L1 Grouping（Bottom-up，L2 段归组到 L1）
  伪L3→L2 Grouping（Bottom-up，伪 L3 段归组到 L2）

取消：
  L3 语义标注（成本高，改为伪 L3 自动生成）
  向上 check（消除）
```

### 待确认细节

- [ ] 伪 L3 最小/最大时长约束（避免切出太短或太长的段）
- [ ] 每个 L2 内伪 L3 段数量范围（建议 3~8 段）
- [ ] Shot boundary 阈值参数
- [ ] 数据引入优先级（HT-Step 最推荐，Shot2Story 需确认 shot 粒度）

---

## 推荐组合

| 目标 | 推荐方向 |
|------|---------|
| 快速扩充数据量 | 方案 A 双轨自动化 + HT-Step / NewsNet 引入 |
| 根本性改变任务 | 方向一（相对粒度）+ 方向四（Grouping 任务） |
| 与现有 reward 兼容 | 方向三最小改动，其次方向四 |
| 直接面向事件关系推理 | 方向四 Grouping 任务 + 方向五 Few-shot 粒度校准 |
