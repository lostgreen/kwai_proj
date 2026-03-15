# YouCook2 Seg Annotation 改造落地方案

本文档给出当前 `proxy_data/youcook2_seg_annotation` 管线向新三层 segmentation 方案迁移的工程化修改方案。目标不是一次性推翻现有代码，而是在保留现有数据可读性的前提下，逐步把标注协议、采样策略、数据构建和下游消费方式切到新的层级定义。

## 1. 当前实现和目标方案的差距

当前仓库的现状：

- L1 已可运行，但本质仍是“整段 clip 的稀疏抽帧 + 宏观阶段 JSON 输出”。
- L2 虽然代码可跑，但它是“按 L1 phase 截帧后再次分段”，还不是 `128s / stride 64s` 的重叠滑窗事件检测。
- L3 目前是“对每个 L2 step 再做 dense chunk 标注”，还不是 `(L2 clip, text query) -> local temporal grounding`。
- 标注 JSON 只保存真实时间轴上的 `start_time/end_time`，没有 warped timeline 映射表，没有 window merge 过程，也没有 grounding query。
- `build_dataset.py` 和奖励函数仍按“单层 segment 边界”思路组织，下游难以直接消费新的三级协议。

你的目标方案对应的真正改造点：

- L1：输入不再等于真实时间轴，而是 `Warped Time Axis`。
- L2：输入不再等于 phase 内一次切完，而是 `128s overlap sliding windows + merge`。
- L3：输入不再等于 dense segmentation，而是 `query-conditioned local temporal grounding`。

## 2. 总体迁移原则

建议采用“协议先行、算法渐进”的迁移方式：

1. 先升级 annotation schema，让 JSON 能表达 warped mapping、window merge、grounding query。
2. 再改 `extract_frames.py / annotate.py / build_dataset.py`，让脚本能写入和读取新字段。
3. 最后再替换 reward / rollout visualization / 训练 prompt，避免一开始就连锁爆炸。

这样做的好处：

- 旧的 `level1/2/3` 文件仍然可读。
- 新旧样本可以在同一个目录共存。
- 训练侧可以先只消费 L2，再逐步打开 L1/L3。

## 3. 新 annotation schema

建议把每个标注文件升级为 `schema_version = 2`，并显式记录每层的输入构造方式。

建议的顶层结构：

```json
{
  "schema_version": 2,
  "clip_key": "GLd3aX16zBg",
  "video_path": "/path/to/full_video.mp4",
  "source_video_path": "/path/to/full_video.mp4",
  "source_mode": "full_video",
  "clip_duration_sec": 241.62,
  "frame_dir": "...",
  "level1": {},
  "level2": {},
  "level3": {},
  "annotated_at": "2026-03-15T00:00:00Z"
}
```

### 3.1 Level 1 schema

L1 不再只存最终 phase，而要同时存 warped 输入是怎么构造出来的。

```json
{
  "input_mode": "warped_time",
  "candidate_spans": [
    {
      "span_id": 1,
      "start_sec": 32.0,
      "end_sec": 118.0,
      "source": "operation_mask_merge",
      "is_cooking_relevant": true
    }
  ],
  "warped_timeline": [
    {
      "warped_index": 1,
      "orig_sec": 32.0,
      "span_id": 1,
      "frame_idx": 33
    }
  ],
  "macro_phases_warped": [
    {
      "phase_id": 1,
      "start_warped_idx": 1,
      "end_warped_idx": 8,
      "phase_name": "Ingredient Preparation",
      "narrative_summary": "Prepare the core ingredients before cooking."
    }
  ],
  "macro_phases_projected": [
    {
      "phase_id": 1,
      "start_sec": 32.0,
      "end_sec": 76.0,
      "phase_name": "Ingredient Preparation",
      "narrative_summary": "Prepare the core ingredients before cooking."
    }
  ]
}
```

核心要求：

- `candidate_spans` 记录被保留的真实时间段，等价于“过滤 talking heads / B-roll 后的可用物理操作区间”。
- `warped_timeline` 是最关键的逆向映射表，后续一切边界恢复都依赖它。
- `macro_phases_warped` 保存模型原始输出。
- `macro_phases_projected` 保存映射回真实时间轴后的最终 GT。

### 3.2 Level 2 schema

L2 不再按单个 phase 一次性切分，而要显式记录滑窗检测和后融合结果。

```json
{
  "input_mode": "overlap_sliding_window",
  "window_size_sec": 128,
  "stride_sec": 64,
  "windows": [
    {
      "window_id": 1,
      "parent_phase_id": 2,
      "start_sec": 64.0,
      "end_sec": 192.0,
      "raw_events": [
        {
          "local_event_id": 1,
          "start_sec": 78.0,
          "end_sec": 111.0,
          "instruction": "Mash the potatoes and mix in seasoning",
          "visual_keywords": ["bowl", "potato", "spoon"]
        }
      ]
    }
  ],
  "merged_events": [
    {
      "event_id": 1,
      "parent_phase_id": 2,
      "start_sec": 78.0,
      "end_sec": 114.0,
      "instruction": "Mash the potatoes and mix in seasoning",
      "visual_keywords": ["bowl", "potato", "spoon"],
      "source_window_ids": [1, 2],
      "merge_method": "interval_fusion"
    }
  ]
}
```

核心要求：

- `windows` 保留原始窗口输出，方便回溯模型错误和调 fusion 参数。
- `merged_events` 才是最终训练标签。
- 每个事件必须满足 aggregation constraint，即“完整逻辑子工序”，不能退化成单个手部动作。

### 3.3 Level 3 schema

L3 改成 query-conditioned grounding，每条记录都要显式绑定查询文本。

```json
{
  "input_mode": "local_temporal_grounding",
  "groundings": [
    {
      "grounding_id": 1,
      "parent_event_id": 3,
      "action_query": "pour the beaten egg into the pan",
      "start_sec": 145.2,
      "end_sec": 147.1,
      "boundary_type": "kinematic",
      "pre_state": "The pan is heated and empty.",
      "post_state": "Liquid egg is now spread across the pan.",
      "allow_gaps": true
    }
  ]
}
```

核心要求：

- `action_query` 是一级公民，不能再隐含在 step 文本里。
- 边界定义从“语义段”切换成“kinematic start/end”。
- 不要求铺满整个 L2 区间，留白是合法的。

## 4. 各脚本的具体修改建议

### 4.1 `extract_frames.py`

目标：支持 L1 的 full-video / full-video-prefix 抽帧，并保留足够元数据供 warped 重建。

建议修改：

- 保留现有 `meta.json`，新增字段：
  - `schema_version`
  - `frame_index_to_sec`
  - `full_video_duration_sec`
  - `annotation_mode`
- 对 full video 抽帧时，默认把 frame index 明确视为 `floor(second) + 1`，统一时间解释。
- 不建议在这一层直接生成 warped frames；这一层只负责提供完整真实时间轴素材。

建议新增 helper：

- `build_frame_time_index(frame_dir) -> list[dict]`
- 输出形如 `[{frame_idx: 1, sec: 0.0, path: "0001.jpg"}]`

### 4.2 `annotate.py`

这是改造的主战场，建议拆成三套 sampler，而不是继续只靠 `frames_for_time_range_to_base64()`。

建议新增函数：

- `build_operation_mask(...)`
  - 输入：全视频稀疏帧
  - 输出：每秒是否属于“有物理操作”的布尔掩码
  - 第一版可直接用 VLM 二分类 prompt，不必先上复杂视觉模型
- `build_warped_timeline(candidate_spans, max_frames_per_call, strategy)`
  - 输出：`warped_timeline` + 对应 frame 文件列表
- `build_overlapping_windows(start_sec, end_sec, window_size=128, stride=64)`
- `merge_window_events(raw_events, iou_thr, text_sim_thr)`
- `build_grounding_queries(event_record)`

具体改法：

- L1：
  - 不再直接对整段 clip 均匀抽帧。
  - 先做 relevance filtering，得到 `candidate_spans`。
  - 再对 `candidate_spans` 做均匀或随机抽帧，构造 warped timeline。
  - Prompt 输出 warped 索引边界，工程端再映射成真实时间。
- L2：
  - 输入切换为 `merged macro phase span -> overlap windows`。
  - 每个 window 单独调用模型，输出该 window 内的完整 event。
  - window 结果全部保存在 `windows`，再跑 NMS / 区间融合得到 `merged_events`。
- L3：
  - 不再要求模型列出所有 micro chunks。
  - 改成对每个 `action_query` 单独做 grounding。
  - query 可来自 `merged_events.instruction` 的 verb-object 抽取，或单独再做一次 query proposal。

### 4.3 `prompts.py`

建议不是简单改文案，而是把 prompt 的“输入协议”写死。

L1 prompt 必须新增：

- 明确告诉模型当前时间轴是 warped，不代表真实时长。
- 输出字段用 `start_warped_idx/end_warped_idx`，不要再让模型直接写真实时间。
- 提醒模型忽略候选 span 之间的真实时间断裂。

L2 prompt 必须新增：

- 当前输入只覆盖一个 `128s` window。
- 只输出 window 内“完整的 logical event”。
- 对跨窗截断事件允许“partial but detectable”，但后处理会融合。

L3 prompt 必须新增：

- 输入是 `(event clip, action_query)`。
- 输出只回答该 query 的 grounding。
- 边界规则必须是物理接触/状态固化，而不是 recipe step 语义。

### 4.4 `build_dataset.py`

建议不要继续把三级任务都塞成同一种 `answer=json.dumps(annotation)`。

更可落地的做法：

- L1 样本：
  - `answer` 只放 `macro_phases_warped`
  - `metadata` 放 `warped_timeline`
- L2 样本：
  - 每个 window 一个训练样本，GT 是该 window 内 `raw_events`
  - 另可额外导出一个 merge 后训练集，GT 为 `merged_events`
- L3 样本：
  - 每个 `(parent_event_id, action_query)` 一条样本
  - `videos` 应切成该 event 对应 clip，而不是整段 full video

建议新增 problem type：

- `temporal_seg_warped_L1`
- `temporal_event_window_L2`
- `temporal_grounding_L3`

### 4.5 Reward 与可视化

当前 [`verl/reward_function/youcook2_temporal_seg_reward.py`](/Users/lostgreen/Desktop/Codes/kwai_proj/verl/reward_function/youcook2_temporal_seg_reward.py) 只支持 `<events>[start,end]` 的单层区间奖励，不够覆盖新协议。

建议按层拆奖励：

- L1 reward：
  - 在 warped 轴上算 boundary F1 或 segment IoU
  - 推理时再根据 `warped_timeline` 映射回真实秒数用于分析
- L2 reward：
  - 对 window 内事件做 current F1-IoU
  - merge 后评估可离线算，不一定放在线 reward
- L3 reward：
  - 直接做单 query 的 temporal IoU / start-end error

可视化侧建议新增：

- L1：同时展示 `warped axis` 和 `projected real axis`
- L2：展示 window raw events 与 merged events 两层轨道
- L3：展示 query、GT grounding、pred grounding 三者对齐

## 5. 最小可落地实施顺序

如果希望尽快产出第一批可用数据，建议按下面顺序走。

### Phase A: 先把 schema 和 L2 改掉

优先级最高，因为 L2 最接近当前 reward 和当前训练接口。

实施项：

1. 给 annotation JSON 加 `schema_version=2`。
2. L2 改成 `128s/64s` overlap windows。
3. 标注文件里保留 `windows` 和 `merged_events`。
4. `build_dataset.py` 先导出 L2 window-level 数据集。

这一步完成后，你已经能拿到最实用的“完整逻辑工序”标注。

### Phase B: 再补 L1 warped timeline

实施项：

1. `extract_frames.py` 默认从 full video 抽帧。
2. `annotate.py` 加 `candidate_spans + warped_timeline` 构造。
3. L1 输出改成 `macro_phases_warped + macro_phases_projected`。

这一步完成后，L1 才真正符合你想要的“打破位置偏见”。

### Phase C: 最后把 L3 切到 grounding

实施项：

1. 设计 `action_query` 生成方式。
2. 以 `(L2 merged event clip, action_query)` 为基本样本。
3. reward 改成 grounding IoU。

这一步完成后，L3 才能稳定服务于 pre_state/post_state 和 RL visual reward。

## 6. 和当前代码的一一对应关系

最需要改的文件：

- [`proxy_data/youcook2_seg_annotation/annotate.py`](/Users/lostgreen/Desktop/Codes/kwai_proj/proxy_data/youcook2_seg_annotation/annotate.py)
  - 当前 L2/L3 主流程都在这里，建议优先重构
- [`proxy_data/youcook2_seg_annotation/prompts.py`](/Users/lostgreen/Desktop/Codes/kwai_proj/proxy_data/youcook2_seg_annotation/prompts.py)
  - 当前 prompt 文本还没把 warped/window/query 当成正式输入协议
- [`proxy_data/youcook2_seg_annotation/build_dataset.py`](/Users/lostgreen/Desktop/Codes/kwai_proj/proxy_data/youcook2_seg_annotation/build_dataset.py)
  - 当前导出格式过于统一，无法充分表达 L1/L2/L3 差异
- [`proxy_data/youcook2_seg_annotation/extract_frames.py`](/Users/lostgreen/Desktop/Codes/kwai_proj/proxy_data/youcook2_seg_annotation/extract_frames.py)
  - 当前 full-video 支持已经有基础，但还缺时间索引和新 schema 元信息
- [`verl/reward_function/youcook2_temporal_seg_reward.py`](/Users/lostgreen/Desktop/Codes/kwai_proj/verl/reward_function/youcook2_temporal_seg_reward.py)
  - 当前 reward 只适配旧式 `<events>` 单层 segmentation

## 7. 我建议你现在就改的版本

如果只选一个最稳的起点，我建议你先落下面这版：

- L1：
  - 暂时不直接训练，只作为全局结构先验
  - 先把 `candidate_spans + warped_timeline + projected_phases` 标进文件
- L2：
  - 先作为主训练任务
  - 用 `128s window / 64s stride / merge` 产出高质量事件锚点
- L3：
  - 先只做少量高质量 query grounding 集，不追求全覆盖

原因很简单：

- L2 对现有训练代码改动最小，但收益最大。
- L1 值得做，但其 reward 和输入协议要多改一层。
- L3 价值最高，但样本成本和协议变化也最大，适合最后接入。

## 8. 一句话结论

当前仓库不需要推倒重来，最合理的改法是：

- 用 `schema_version=2` 先把新三层协议装进去；
- 用 `annotate.py` 新增 `warped sampler / sliding windows / grounding query` 三个构造器；
- 先把 L2 作为主线产数据，再逐步补齐 L1 warped 和 L3 grounding。
