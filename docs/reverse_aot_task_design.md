# 基于当前 code base 的倒放任务设计方案

## 1. 背景和目标

你提到的论文是 `Seeing the Arrow of Time in Large Multimodal Models`。论文里和我们最相关的点，不是简单“把视频倒放喂进去”，而是把 **时间方向性** 单独做成训练信号，让模型学会区分：

1. 这个视频是正放还是倒放
2. 正放和倒放对应的语义描述应该不同
3. 需要时间方向才能答对的问题，不应该被模型“无脑静态化”

论文里的 AoTBench 主要有 3 类任务：

- `Sequence Direction Classification`: 判断 forward / reverse
- `Directional Caption Matching`: 判断视频和描述的方向是否一致
- `AoT-sensitive VQA`: 需要时间方向才能答对的 QA

结合当前仓库，我建议先不要一上来照搬整套 ArrowRL，而是先做一版 **基于现有 YouCook2 数据可直接合成的 AoT/倒放任务集**。不过这里要先区分两种完全不同的“反向”：

1. **播放方向反转**
   - 针对一段连续视频
   - 通过帧序反转得到真正的 reverse playback
   - 这是论文里最直接的 AoT 信号

2. **事件顺序反转**
   - 针对多段 event clip 序列
   - 只调整 clip 的排列顺序
   - 每个 clip 内部仍保持正常正放
   - 这更像“时序顺序建模”，不等价于真正的视频倒放

这两个信号不能混在一起讨论。基于你现在的反馈，第一阶段应该更聚焦在第一种，也就是 **真正的 forward / reverse 判别**。原因很简单：

- 当前仓库已经有成熟的 `problem_type -> reward dispatch -> homogeneous batching` 框架
- 当前数据本身就带时序结构，尤其是 YouCook2 event clips 和 segmentation 窗口
- 先把“倒放任务本身”跑起来，性价比高，也更容易定位收益
- 论文里的 `reverse reward` 需要 trainer 级改造，适合第二阶段做

## 2. 当前仓库里已经可复用的数据和机制

### 2.1 数据现状

当前仓库里和 AoT 最相关的数据有三块：

1. `proxy_data/proxy_train_easyr1.jsonl`
   - 共 `5332` 条
   - 任务分布：`add=1333, delete=1333, replace=1333, sort=1333`
   - 本质上是基于 YouCook2 event clip 构造的离散时序代理任务

2. `proxy_data/youcook2_train_easyr1.jsonl`
   - 共 `4463` 条
   - 每条是一个较长 cooking clip
   - `metadata` 里已经有：
     - `segments`
     - `sentences`
     - `clip_duration`
     - `num_events`
   - 这批数据特别适合做“方向感知描述匹配”

3. `proxy_data/proxy_train_text_options.jsonl`
   - 共 `1000` 条
   - 只有 `add` / `replace`
   - 选项从视频改成了文本 sentence
   - 很适合做轻量版 reverse-text matching

### 2.2 训练和 reward 现状

当前框架已经具备以下条件：

- `problem_type` 驱动的任务分桶采样
- `mixed_proxy_reward.py` 统一 reward 入口
- `add/delete/replace` 复用 `_choice_reward`
- `sort` 复用 `_sort_reward`
- `temporal_seg` 复用 `_temporal_seg_reward`
- `TaskHomogeneousBatchSampler` 已支持多任务权重采样

这意味着：**只要我们能把新倒放任务写成新的 `problem_type`，并保持答案是字母或序列，现有训练链路几乎不用大改。**

## 3. 设计原则

我建议按下面 5 条原则设计倒放任务：

1. 优先复用现有标注，不引入额外人工标注
2. 优先做 MCQ 或排序型任务，直接复用现有 reward
3. 优先保证方向信号强，不要把大量弱时间性样本混进来
4. 严格区分“帧级倒放”和“事件级重排”，不要把两者误写成同一种 reverse
5. 把“倒放样本构造”与“ArrowRL reverse reward”分阶段处理

## 4. 推荐的任务集合

我建议分成“主线任务”和“备选任务”。

### 4.1 主线任务：`reverse_cls`

目标：
判断一个视频片段是正放还是倒放。

数据来源：

- 优先用 `youcook2_train_easyr1.jsonl` 的长 clip
- 次选 `proxy_train_easyr1.jsonl` 中的 event clip

更推荐的优先级：

- 第一优先级：`youcook2_train_easyr1.jsonl` 的连续 window clip
- 第二优先级：长度更长、动作更完整的 event clip 序列拼接片段

不太建议一开始就用特别短的单个 event clip，因为太短的 clip 很多时候方向性不够强。

样本构造：

- 对同一个 clip，随机采样两种播放方式：
  - 正放
  - 倒放
- 这里的“倒放”指 **帧序反转**
- 也就是说，这个任务里确实需要把视频内部帧顺序 reverse 掉
- prompt 为二选一选择题，例如：

```text
Watch the cooking video clip carefully.
Is this clip played in the normal forward direction or in reverse?

Options:
A. forward
B. reverse

Think in <think> and answer in <answer>.
```

标签：

- 正放样本答案为 `A`
- 倒放样本答案为 `B`

reward：

- 直接复用 `_choice_reward`

为什么值得先做：

- 最贴近论文的 Sequence Direction Classification
- 不依赖额外文本生成
- 是最纯粹的 AoT 监督信号

### 4.2 备选任务：事件顺序一致性任务

这一类任务不是严格意义上的“视频倒放任务”，而是“事件顺序理解任务”。如果后面发现 `reverse_cls` 单任务过窄，再考虑加它们。

#### 任务 B: `order_v2t`（备选）

目标：
给定一串 event clips，判断哪段文本顺序和 clip 顺序一致。

这里要非常明确：

- 如果底层样本来自 `metadata.sentences`
- 那么所谓“reverse text”应理解为 **事件顺序反转**
- 不应该把每个 event clip 内部也做帧级倒放

也就是说，这个任务的视觉侧应该是：

- `clip_1, clip_2, clip_3`
  或
- `clip_3, clip_2, clip_1`

其中每个 `clip_i` 自己内部都还是正常正放。

对应文本侧才是：

- forward-order text
- reverse-order text

这个任务本质是 order matching，不是 playback direction classification。

为什么它可以作为备选：

- 能利用现有 `metadata.sentences`
- 工程成本不高

为什么它不该是第一优先级：

- 它学的是“事件排列”
- 不一定真的提升对单段视频内部 AoT 的感知
- 和论文里最核心的 reverse playback 信号不完全一样

#### `rev_add` / `rev_replace` / `rev_sort`（暂不建议）

这部分我认同你的担心，第一版确实不建议主推。

原因有 3 点：

1. 它们更像“在逆时间轴上重新定义任务”，而不是真正在学 forward/reverse 感知
2. 它们可能让模型学到数据构造规则，而不是 AoT 本身
3. 即便有效，也很难解释收益到底来自“时间方向”还是“任务变难了”

所以更合理的定位应该是：

- 可以作为以后的小规模消融实验
- 不应该作为当前 AoT 主设计

#### `reverse_t2v` / `reverse_temporal_seg`

这两类先继续放在后面：

- `reverse_t2v` 训练成本太高
- `reverse_temporal_seg` AoT 监督太弱

都不建议进第一轮设计

## 5. 最推荐的数据实现方式

这里有一个非常关键的实现结论：

**不同任务，对“反向”的实现方式应该不同。**

### 5.1 对 `reverse_cls`

建议：

- 不改 loader 去“在线倒放视频”
- 优先离线把倒放样本写成 frame-list
- 对同一段连续 clip，把帧列表做 `frames[::-1]`

这是标准的 playback reverse。

### 5.2 对 `order_v2t` 这类事件顺序任务

建议：

- 不做 clip 内部帧倒放
- 只调整多段 event clip 的排列顺序

例如：

- forward sequence: `[clip1, clip2, clip3]`
- reverse-order sequence: `[clip3, clip2, clip1]`

但 `clip1/clip2/clip3` 内部帧顺序保持不变。

这才和 `sentences[::-1]` 是严格对齐的。

原因：

- `verl/utils/dataset.py` 已经支持 `videos` 字段既可以是视频路径，也可以是 `list[list[str]]` 的帧路径列表
- `local_scripts/extract_video_frames_to_jsonl.py` 已经证明这条链路能跑
- 对于真正的 playback reverse 样本，我们不需要真的生成新的 mp4，只需要把帧路径顺序反过来即可

也就是说，最稳的 MVP 是：

1. 先把源视频样本离线抽帧
2. `reverse_cls` 对应样本，把帧列表做 `frames[::-1]`
3. 如果后续做 `order_v2t`，则只调整 clip list 顺序，不改 clip 内部帧序
4. 直接写回新的 JSONL

这样做的好处：

- 不改训练 loader
- 不依赖 ffmpeg reverse 重新编码
- 数据可复现，容易 debug
- forward / reverse 可以共享同一套原始帧文件

## 6. 建议的数据构造脚本

我建议新增一个脚本：

- `proxy_data/build_reverse_aot_tasks.py`

它负责 4 件事：

1. 读取源数据
   - `proxy_train_easyr1.jsonl`
   - `youcook2_train_easyr1.jsonl`
   - 可选 `proxy_train_text_options.jsonl`

2. 生成倒放帧视图
   - 若样本还是原始 mp4，可先离线转成 frame-list JSONL
   - 对 reverse 样本直接把帧顺序反转

3. 生成新任务样本
   - `reverse_cls`
   - 可选 `order_v2t`

4. 写出可直接训练的 JSONL

建议输出文件：

- `proxy_data/reverse_aot_train.jsonl`
- `proxy_data/reverse_aot_val.jsonl`

## 7. 任务字段设计

建议统一保留下面这些字段：

```json
{
  "messages": [{"role": "user", "content": "..."}],
  "prompt": "...",
  "answer": "A",
  "videos": [["frame_0001.jpg", "frame_0002.jpg", "..."]],
  "data_type": "video",
  "problem_type": "reverse_cls",
  "metadata": {
    "source_task": "temporal_seg",
    "video_id": "...",
    "is_reversed": true,
    "clip_duration": 84,
    "num_events": 5
  }
}
```

对于 `order_v2t`，建议额外记录：

```json
{
  "metadata": {
    "source_task": "temporal_seg",
    "forward_sentences": ["...", "...", "..."],
    "reverse_sentences": ["...", "...", "..."],
    "reverse_mode": "clip_order_only"
  }
}
```

## 8. Reward 接入建议

第一阶段尽量不新增复杂 reward，直接沿用现有逻辑：

- `reverse_cls` -> `_choice_reward`
- 可选 `order_v2t` -> `_choice_reward`

因此只需要在 `verl/reward_function/mixed_proxy_reward.py` 的 `_TASK_REWARD_DISPATCH` 里补几项映射即可。

建议新增：

```python
{
    "reverse_cls": _choice_reward,
    "order_v2t": _choice_reward,
}
```

## 9. 混合训练怎么并进当前框架

### 9.1 如果你想走最小改动路径

直接新增一个 AoT 数据文件，然后并进现有 mixed dataset：

- 原始：`add / delete / replace / sort / temporal_seg`
- 新增：`reverse_cls`
- 可选新增：`order_v2t`

建议初始权重不要太激进，先让 AoT 任务占 `10%` 到 `20%`：

```json
{
  "temporal_seg": 0.35,
  "add": 0.12,
  "delete": 0.12,
  "replace": 0.12,
  "sort": 0.12,
  "reverse_cls": 0.12,
  "order_v2t": 0.05
}
```

如果第一轮只做最干净的方案，那就只加 `reverse_cls`，连 `order_v2t` 都可以先不加。

### 9.2 如果你想先贴合当前 `run_mixed_proxy_dapo.sh`

当前脚本实际上只用了：

- `proxy_train_text_options.jsonl`
- 任务只有 `add` / `replace`

那么可以先做一个更轻量的版本：

- `reverse_cls`

如果后面要加文本侧任务，再考虑：

- `order_v2t`

但不建议在当前阶段继续推 `rev_add_text` / `rev_replace_text`。

## 10. 样本筛选建议

倒放任务最大的问题，不是构造不出来，而是很多样本 **方向信号太弱**。如果把这些弱样本大量加入训练，收益可能不明显。

我建议至少做下面几条过滤：

1. `clip_duration >= 6s`
2. `num_events >= 3` 的窗口优先
3. 优先保留 `metadata.is_fallback = false` 的 segmentation 样本
4. 对 `order_v2t`，只保留 `sentences` 数量 >= 3 的窗口
5. 对 `reverse_cls`，优先选择包含明显手部操作、倒入、翻面、点火、切分等 cooking 动作的样本

如果后面想更接近论文，可以再做一轮基于模型输出的样本筛选：

- 比较 forward / reverse 下首 token 分布差异
- 用近似 TDS 过滤出高时间敏感样本

但这属于第二阶段，不是 MVP 必需项。

## 11. 第二阶段：再考虑论文式 reverse reward

论文最核心的不是只有数据倒放，而是：

`总 reward = fidelity reward + alpha * reverse reward`

其中 reverse reward 逼模型满足：

- 正放视频的回答，应该和倒放视频的回答拉开
- 如果一个样本本身时间方向不敏感，则动态关闭这项 reward

这件事在当前仓库里 **不能只靠 reward function 文件完成**，因为它需要：

1. 对同一样本的 reversed video 再跑一遍模型，拿到 `o_rev`
2. 用 `Similarity(o_i, o_rev)` 计算额外奖励
3. 最好再做 `alpha` / `gamma` 的动态门控

这意味着要改 trainer 或 rollout 逻辑，而不是只改 `mixed_proxy_reward.py`。

所以我建议：

- 第一阶段：先把 reverse tasks 数据集和普通 fidelity reward 跑起来
- 第二阶段：如果 reverse tasks 明显有效，再上 trainer 级 reverse reward

## 12. 最终推荐方案

如果目标是“尽快验证论文思路对当前项目是否有效”，我建议按下面顺序推进：

### Phase 1: 最小可行版本

只做这 1 类任务：

- `reverse_cls`

实现方式：

- 用 frame-list 离线表示 reverse
- 不改 loader
- 只扩展数据脚本和 reward dispatch
- 样本优先来自连续 clip，而不是短 event clip

### Phase 2: 小规模混训验证

把 `reverse_cls` 按 `10%` 到 `15%` 左右权重并进 mixed proxy 训练，重点看：

- `reward/reverse_cls`
- 原 `add/replace/sort` 是否同步提升
- 验证集 forward / reverse 的差异是否变大

如果这一步确实有收益，再考虑补一个弱一些的顺序任务：

- `order_v2t`

### Phase 3: trainer 级 ArrowRL

如果前两阶段有正收益，再做：

- reverse response 辅助 rollout
- `fidelity + reverse reward`
- 动态门控 `alpha/gamma`

## 13. 一句话结论

结合当前代码库，最值得先做的是一个非常干净的 `reverse_cls`：对连续视频做真正的 forward / reverse 判别。至于基于 `sentences` 的任务，如果要做，应该被定义成“事件顺序匹配”而不是“clip 内部倒放”。`rev_add/rev_replace/rev_sort` 这类逆世界任务暂时不建议作为主设计，因为它们更像人为改写任务，而不一定真的在教模型理解时间方向。
