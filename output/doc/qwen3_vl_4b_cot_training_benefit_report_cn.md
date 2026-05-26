# Qwen3-VL-4B CoT 训练收益与能力分析汇报

日期：2026-05-20

## 1. 结论先行

本次分析覆盖三个已经完成的 100-step CoT teacher run：

- `qwen3_vl_4b_aot_100step_cot`
- `qwen3_vl_4b_seg_100step_cot`
- `qwen3_vl_4b_logic_100step_cot`

核心结论：

1. CoT 训练在 100 step 内带来了明显的验证集收益。三个 run 的 val reward 均从约 0.27-0.33 提升到 0.53-0.58，其中 Logic 最强，Seg 次之，AOT 有收益但不够稳定。
2. 模型确实学到了“可用的短 CoT 格式”。到 step100，Seg 和 Logic 的 CoT format reward 接近 1.0，repair ratio 低于 1%。这说明它们不是只会输出空壳标签，而是能够稳定地产生 `<thought>...</thought>` 并在预算内给出答案。
3. 从 case 看，CoT 多数是任务相关的视觉和时间证据，而不是纯格式 farming。高分 case 会引用具体视觉事件、时间范围、场景切换或因果顺序；低分但格式正确的 case 通常是“看错/判断错”，不是空泛模板。
4. TG 任务的 CoT 有真实能力信号：模型会从粗略描述逐步变成明确的时间定位依据，例如“polo match starts around 22s and continues to 50s”。但 TG 也暴露了一个问题：有些 case 的 CoT 合理、格式完整，最终时间边界仍会漂移，说明 CoT 可解释性不等于定位完全可靠。
5. AOT 是主要风险点。AOT proxy task 的 CoT 经常过长并触发 repair，说明 128-token thought budget 对 AOT 对比式任务偏紧。AOT 有学到任务行为，但 CoT 预算纪律不足。

需要谨慎表述的点：

- 当前报告能支持“CoT 训练过程中模型能力和 CoT 质量提升”的 claim。
- 若要严格 claim“CoT 相对 no-CoT 有多少净收益”，还需要同模型、同数据、同 step、同 decoding 设置的 no-CoT 对照 run。现有数据更适合证明 CoT teacher run 自身的学习曲线和 CoT 能力，而不是做严格 ablation。

## 2. 训练收益概览

通用设置：

- `MAX_STEPS=100`
- `VAL_FREQ=25`
- `MAX_RESPONSE_LEN=256`
- `COT_BUDGET_MAX_TOKENS=128`
- CoT tag：`<thought>...</thought>`

### 2.1 Val reward 提升

| Run | Val reward 0 -> 100 | 绝对提升 | 观察 |
| --- | ---: | ---: | --- |
| AOT | 0.3231 -> 0.5292 | +0.2061 | 有明显收益，但 step50 后有波动 |
| Seg | 0.2708 -> 0.5452 | +0.2744 | 提升稳定，step100 继续变好 |
| Logic | 0.3269 -> 0.5765 | +0.2496 | 最强，最终 val 最高 |

学习曲线：

| Run | Step 0 | Step 25 | Step 50 | Step 75 | Step 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AOT | 0.3231 | 0.5257 | 0.5437 | 0.5279 | 0.5292 |
| Seg | 0.2708 | 0.4805 | 0.5250 | 0.5392 | 0.5452 |
| Logic | 0.3269 | 0.5346 | 0.5634 | 0.5756 | 0.5765 |

汇报 claim：

> 在 100-step CoT teacher training 内，三个 run 的验证集 reward 都出现了实质提升，说明训练不是只在优化格式，而是同步改善了任务表现。

### 2.2 CoT 格式和预算稳定性

| Run | CoT format 0 -> 100 | Val100 start | Val100 end | Val100 repair | Val100 thought words mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| AOT | 0.7563 -> 0.9137 | 1.000 | 0.982 | 0.173 | 68.2 |
| Seg | 0.7430 -> 0.9963 | 1.000 | 0.997 | 0.007 | 50.8 |
| Logic | 0.7657 -> 0.9953 | 1.000 | 0.998 | 0.009 | 48.4 |

汇报 claim：

> Seg 和 Logic 已经学会了稳定、短、格式正确的 CoT。AOT 也明显改善，但 task-specific AOT case 中 repair ratio 偏高，说明 128-token 预算对 AOT 的 clip/paragraph 对比任务比较紧。

## 3. 模型是否真的学到了 CoT 能力

判断标准不是“有没有 `<thought>` 标签”，而是看 thought 是否包含任务相关证据：

- 是否引用视觉内容：物体、动作、场景、文字、镜头变化；
- 是否引用时间信息：start/end、around 22s、from 0:00 to 0:02；
- 是否解释答案选择：为什么选 A/B/C，为什么不是其他选项；
- 是否能随着训练从空泛描述变得更具体、更短、更稳定；
- 错误 case 是否仍然是“有依据但判断错”，而不是纯模板。

整体判断：

- Logic：最像真正学到了 CoT。它用较短 thought 做事件顺序和因果推理，最终 reward 最高、KL 最低、repair 很少。
- Seg：也学到了有效 CoT，尤其 L1/L2 分割中会用“场景/活动变化”解释边界。L3 仍偏粗，主要是 granularity 问题。
- AOT：学到了对比式 reasoning，但经常写太长。AOT 的失败更像“预算和视觉判别双重压力”，不是没有 CoT。

## 4. TG 任务专项分析

TG 是很好的 CoT 能力观察窗口，因为答案要求时间定位，thought 应该解释“为什么这个时间段对应目标事件”。

### 4.1 TG task-level 指标

| Run | TG n | Val100 TG reward | CoT ok | Repair | Avg thought words |
| --- | ---: | ---: | ---: | ---: | ---: |
| AOT | 600 | 0.2930 | 0.9283 | 0.0717 | 49.0 |
| Seg | 600 | 0.3020 | 0.9950 | 0.0050 | 40.2 |
| Logic | 600 | 0.3011 | 0.9983 | 0.0017 | 36.0 |

TG 的结论比较克制：

- 三个 run 的 TG final reward 接近 0.29-0.30，不是特别高；
- 但 TG 的 CoT 格式质量很好，尤其 Seg/Logic 几乎不 repair；
- TG 的主要瓶颈不是“不愿意写 CoT”，而是时间边界定位仍会漂移；
- 好的 TG CoT 能提供可检查的 evidence，例如“polo match around 22-50s”，这对汇报展示很有价值。

### 4.2 Val 子类拆解：TG 提升是否弱于其他任务

先看 TG 自身。三个 run 的 TG 都从约 `0.15` 涨到约 `0.29-0.30`，绝对提升约 `+0.14` 到 `+0.15`。这说明 CoT run 内 TG 不是完全没学，但它在 step50 后基本进入平台期，最终绝对分仍低。

| Run | TG step0 | step25 | step50 | step75 | step100 | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AOT | 0.1545 | 0.2551 | 0.2917 | 0.2956 | 0.2930 | +0.1385 |
| Seg | 0.1523 | 0.2516 | 0.2891 | 0.2960 | 0.3020 | +0.1497 |
| Logic | 0.1513 | 0.2484 | 0.2900 | 0.3030 | 0.3011 | +0.1498 |

所以如果和你观察到的 direct training 相比，direct TG 能提升约 5 个点、CoT 只能提升约 1 个点，这个现象是合理的：**TG 可能不是 CoT 最受益的任务**。CoT 能帮助模型写出可解释 evidence，但 precise boundary 仍需要更直接的时间定位监督。

再看其他 val 子类。MCQ、Seg proxy、Logic proxy 的提升更明显：

| Run | Task | step0 | step100 | delta | 结论 |
| --- | --- | ---: | ---: | ---: | --- |
| AOT | llava_mcq | 0.4467 | 0.7792 | +0.3325 | base MCQ 提升很大 |
| AOT | temporal_grounding | 0.1545 | 0.2930 | +0.1385 | TG 提升较小，50 后平台 |
| AOT | seg_aot_action_t2v_binary | 0.4122 | 0.4966 | +0.0845 | AOT proxy 不稳定 |
| AOT | seg_aot_event_t2v_binary | 0.4514 | 0.6181 | +0.1667 | 有提升但波动 |
| Seg | llava_mcq | 0.4517 | 0.8175 | +0.3658 | 提升很大 |
| Seg | temporal_grounding | 0.1523 | 0.3020 | +0.1497 | TG 中等提升 |
| Seg | temporal_seg_hier_L1 | 0.0334 | 0.4831 | +0.4497 | proxy 提升很强 |
| Seg | temporal_seg_hier_L2 | 0.0137 | 0.5079 | +0.4941 | proxy 提升很强 |
| Seg | temporal_seg_hier_L3_seg | 0.0219 | 0.3233 | +0.3014 | proxy 提升明显但仍偏难 |
| Logic | llava_mcq | 0.4508 | 0.8150 | +0.3642 | 提升很大 |
| Logic | temporal_grounding | 0.1513 | 0.3011 | +0.1498 | TG 中等提升 |
| Logic | event_logic_fill_blank | 0.4384 | 0.6630 | +0.2246 | proxy 提升明显 |
| Logic | event_logic_predict_next | 0.5161 | 0.6371 | +0.1210 | 起点较高，提升较小 |
| Logic | event_logic_sort | 0.1217 | 0.6458 | +0.5241 | proxy 提升最强 |

分类判断：

- **TG：** 三个 run 都提升约 `+0.14/+0.15`，但最终只有 `0.30` 左右，而且 step50 后基本持平。它确实是 CoT 效果相对不突出的任务。
- **MCQ：** 提升最稳，三个 run 都有约 `+0.33/+0.36`。这说明 base visual QA 能明显受益于训练。
- **Seg proxy：** L1/L2 提升非常强，L3 也提升但绝对分仍低。Seg proxy 的主要问题不是 CoT 格式，而是 L3 granularity。
- **Logic proxy：** `event_logic_sort` 提升最大，`fill_blank` 也明显；`predict_next` 起点较高，delta 较小但最终仍不错。
- **AOT proxy：** 最不稳定。部分子类有提升，但 action v2t 几乎没涨，且 AOT proxy 的 repair ratio 高，说明 CoT 预算压力确实比较严重。

这一节的汇报口径可以是：

> CoT 训练对不同任务的收益不均匀。MCQ、Seg L1/L2、Logic sort/fill_blank 的 val 提升明显；TG 虽然从 0.15 提升到 0.30 左右，但 step50 后基本平台，且最终绝对分偏低。结合 case 观察，TG 的问题不是 CoT 完全无效，而是模型能写出事件证据，却经常无法稳定转换成精确 start/end 边界。因此 TG 可能更需要 direct temporal boundary supervision 或专门的边界损失/格式约束，而不是单靠通用 CoT。

### 4.3 TG Case A：从模糊定位到精确时间段

Run：`logic`  
Task：`temporal_grounding`  
Case：`v_VFqkLp5mzBM`  
GT：`20.58 - 50.66 seconds`  
Reward：`0.0000 -> 0.9309`

逐阶段 CoT：

- Step 0，reward `0.0000`，repair `False`，answer：`0:22 - 0:51 seconds`。  
  CoT：The video shows a polo match with horses starting around **0:22**, where players on horseback are seen racing and chasing a ball across a grassy field. The scenes clearly depict horses in motion, indicating the event of a polo match with horses.

- Step 25，reward `0.9309`，repair `False`，answer：**`21.3 - 51.5 seconds`**。  
  CoT：The video shows a polo match with horses in motion, players on horseback chasing a ball, and spectators watching. This is evident from the frames around **0:21 to 0:51**, where the polo match is actively taking place with horses and riders.

- Step 50，reward `0.8925`，repair `False`，answer：**`22.0 - 51.7 seconds`**。  
  CoT：The polo match with horses is shown in the video from **22.0 - 51.7 seconds**. This segment features riders on horses, actively playing polo on a green field, with spectators and cones visible in the background.

- Step 75，reward `0.8977`，repair `False`，answer：**`22.5 - 50.3 seconds`**。  
  CoT：The polo match with horses is shown starting from **22.5 seconds**, where players on horseback are seen actively playing the game, chasing and hitting the ball.

- Step 100，reward `0.8872`，repair `False`，answer：**`22.5 - 51.3 seconds`**。  
  CoT：The polo match with horses is shown starting from **22.5 seconds**, where players on horseback are seen actively playing the game, chasing and hitting the ball with mallets. This segment clearly depicts the event described.

分析：

这是强正例。模型没有只说“这个事件发生了”，而是给出视觉证据和时间范围：骑手、马、追球、草地、观众。CoT 与 reward 提升一致，说明模型学到了一种可解释的 temporal grounding 策略。

### 4.4 TG Case B：CoT 变短但保持证据

Run：`seg`  
Task：`temporal_grounding`  
Case：`v_sPSfixKrDc0`  
GT：`26.16 - 96.77 seconds`  
Reward：`0.0000 -> 0.9355`

逐阶段 CoT：

- Step 0，reward `0.0000`，repair `False`，answer：**`0:27 - 0:55 seconds`，end 过早**。  
  CoT：The video transitions from the man speaking at the counter to him working on the truck cake. At **0:27**, he is seen at the counter with various supplies, and by **0:55**, the cake assembly is underway. The sped-up motion is evident as he works quickly, assembling the truck cake, which was previously shown in the video. This sequence matches the description of the man assembling the large truck cake.

- Step 25，reward `0.9087`，repair `False`，answer：**`28.5 - 98.8 seconds`**。  
  CoT：The video transitions from the man speaking at the counter to him working on the truck cake. The sped-up motion begins around **0:28**, showing him assembling the cake with various supplies, which matches the description of the event.

- Step 50，reward `0.0857`，repair `False`，answer：**`28.0 - 39.0 seconds`，把 1:39 写成 39.0，边界崩**。  
  CoT：The video transitions from the man speaking to the camera, then shows him working on assembling the truck cake. The sped-up motion begins after the initial setup, showing him using various supplies to assemble the large truck cake. This sequence is evident from **0:28 to 1:39**, where the man is actively engaged in the assembly process.

- Step 75，reward `0.9355`，repair `False`，answer：**`27.0 - 99.0 seconds`**。  
  CoT：The video transitions from the man speaking at the counter to him working on the truck cake. The sped-up motion begins around **27 seconds**, showing him assembling the cake with various supplies. This matches the description of the event.

- Step 100，reward `0.7782`，repair `False`，answer：**`26.3 - 108.5 seconds`，start 对齐但 end 偏长**。  
  CoT：The video transitions from the man speaking at the counter to him working on a large truck cake. The sped-up motion shows him assembling the cake with various supplies, which matches the description of the event.

分析：

这是“学到了，但还会抖”的例子。CoT 一直围绕正确事件：man speaking -> truck cake assembly -> sped-up motion。reward 的波动说明 TG 的最终时间边界仍敏感，但 thought 本身是任务相关的，不是格式空壳。

### 4.5 TG Case C：视觉证据正确，但预算或边界导致失败

Run：`aot`  
Task：`temporal_grounding`  
Case：`daclapuOQnE`  
GT：`73 - 82 seconds`  
Reward：`0.0000 -> 0.9209 -> 0.0000`

逐阶段 CoT：

- Step 0，reward `0.0000`，repair `False`，answer：**`1:07 - 1:10 seconds`，偏早**。  
  CoT：At **1:07**, the person picks up the scrubber and then at **1:10**, they spray water onto the chrome trim. This aligns with the action described in the sentence.

- Step 25，reward `0.0000`，repair `False`，answer：**`1:11 - 1:16 seconds`，仍偏**。  
  CoT：The event 'take scrub and spray water' involves the person picking up the scrubber and spraying water on it. This is seen in the video around **1:11-1:16**, where the person holds the scrubber and then sprays water onto it.

- Step 50，reward `0.9209`，repair `False`，answer：**`72.7 - 81.6 seconds`，和 GT 对齐**。  
  CoT：The event involves taking a scrub and spraying water, which is shown when the person picks up the scrubber and then sprays water onto it, preparing it for use. This action is clearly visible in the video between **72.7 and 81.6 seconds**.

- Step 75，reward `0.0000`，repair **`True`**，answer：**`1.1 - 14.9 seconds`，仍有 answer 但时间解析/边界崩**。  
  CoT：1. The video shows a person preparing to clean a car's chrome trim. At **0:08**, a spray bottle and a container of "Karpal" are visible, indicating cleaning products are ready. 2. The person then picks up a small scrubber or pad at **0:46**, which they use to scrub the chrome surface. 3. At **1:11**, the person sprays water onto the chrome trim, which is consistent with the action of spraying water after scrubbing. 4. The action of spraying water is clearly depicted from **1:11 to 1:14,** 

- Step 100，reward `0.0000`，repair **`True`**，answer：**`57.0 - 60.5 seconds`，偏到错误区间**。  
  CoT：1. The video shows a person preparing to clean a car's chrome trim. At **0:08**, a spray bottle and a container of "Karpal" are visible, indicating cleaning products are ready. 2. The person then picks up a small scrubber or pad at **0:45**, which they use to scrub the chrome surface. 3. At **0:57**, the person sprays water onto the chrome trim, which is consistent with the action of spraying water after scrubbing. 4. The action of spraying water is clearly depicted from **0:57 to 1:00,**

分析：

这是很适合汇报的反例：step50 的 CoT 非常好，也直接对应高 reward；但后续 step75/100 thought 变成长，开始描述过多前置准备，触发 repair。注意 repair 后并不是没有 answer，而是**仍给出 answer，但答案边界偏到错误区间**。它说明模型确实能学到 TG CoT，但 CoT budget 和“只写关键证据”的纪律仍然重要。

### 4.6 TG Case D：CoT 合理但最终定位仍不稳

Run：`logic`  
Task：`temporal_grounding`  
Case：`v_ZhUC4qTGdHY`  
GT：`145.39 - 164.28 seconds`  
Reward：`0.0000 -> 0.9186`

逐阶段 CoT：

- Step 0，reward `0.0000`，repair `False`，answer：**`2:26 - 2:44 seconds`，起点晚**。  
  CoT：The video transitions to black screens with text at **2:26**, displaying "Special thanks" followed by "Tignes - Coreupt.com", then "Music: MASS HYSTERIA 'MH 2.C.E.S.'", then "TMP RACING 06 84 82 92 62", and finally "www.a3fvideo.com". This sequence of text on black screens constitutes the ending title screens, which concludes the video.

- Step 25，reward `0.2826`，repair `False`，answer：**`158.4 - 164.2 seconds`，只截到后半段**。  
  CoT：The video transitions to black screens with text credits, including "Special thanks", "Music: MASS HYSTERIA", "TMP RACING", and the website "www.a3fvideo.com". This sequence is the ending title screen, which appears after the main content.

- Step 50，reward `0.2051`，repair `False`，answer：**`159.9 - 164.8 seconds`，仍只截后半段**。  
  CoT：The video transitions to black screens with text credits, including special thanks, music credits, and contact information for TMP Racing and the video production company. This sequence of text screens concludes the video, marking the end of the content.

- Step 75，reward `0.9186`，repair `False`，answer：**`146.6 - 164.1 seconds`，与 GT 对齐**。  
  CoT：The video transitions to black screens with text credits, including special thanks, music credits, and contact information for TMP Racing and the video production company. This sequence of text screens marks the end of the video content.

- Step 100，reward `0.9186`，repair `False`，answer：**`146.6 - 164.1 seconds`，保持稳定**。  
  CoT：The video transitions to black screens with text credits, including special thanks, music credits, and contact information for TMP Racing and the video production company. This sequence of text screens marks the end of the video content.

分析：

这个 case 说明 TG CoT 的能力不是只来自动作识别，也能识别文本/片尾结构。它支持“模型学到了 temporal evidence extraction”，但早期 step 的低 reward 也说明“识别事件类型”和“精确定位边界”是两件事。

## 5. 非 TG 代表 case：CoT 是否真实

### 5.1 Seg L2：从整段输出到正确切分

Run：`seg`  
Task：`temporal_seg_hier_L2`  
Case：`TVbcskkKIJA`  
GT：`<events>[[0, 88], [88, 171]]</events>`  
Reward：`0.0000 -> 1.0000`

逐阶段 CoT：

- Step 0，reward `0.0000`，repair **`True`**，answer：**空，thought 被截断**。  
  CoT：The video starts with the man speaking to the camera in an outdoor setting (0-22s). Then, the camera shifts to a close-up of the car's undercarriage, where the man begins to work on the wheel hub assembly (22-28s). He continues manipulating the parts, and the camera remains focused on the same area (28-34s). Finally, the man is seen adjusting or removing a component from the wheel hub, which is the final task in this sequence (34-42s). These are distinct tasks within the same location and subject, so they are merged into

- Step 25，reward `0.3431`，repair `False`，answer：**`<events>[[0, 171]]</events>`，under-seg**。  
  CoT：The video starts with the man speaking to the camera in an outdoor setting, then transitions to a close-up of a car's undercarriage where he begins working on the wheel hub. The camera focuses on his hands manipulating the components, showing detailed work. The man continues to adjust and inspect the parts, maintaining focus on the task. There are no significant scene changes or shifts in location or subject, indicating that this is **one continuous task**. The entire segment from 0 to 171 seconds is dedicated to this mechanical work, with the camera moving slightly to capture different angles of the same task.

- Step 50，reward `0.3431`，repair `False`，answer：**`<events>[[0, 171]]</events>`，仍 under-seg**。  
  CoT：The video starts with the man speaking to the camera in an outdoor setting, then transitions to a close-up of a car's undercarriage where he begins working on a mechanical component. The focus shifts from the man speaking to the detailed work on the car part, indicating a change in subject and task. The man's hands are shown manipulating the component, suggesting a continuous task. The entire sequence from 0 to 171 seconds is a **single continuous task** of explaining and demonstrating car maintenance.

- Step 75，reward `0.9884`，repair `False`，answer：**`<events>[[0, 89], [89, 171]]</events>`，接近 GT**。  
  CoT：The video starts with the man speaking outdoors, then transitions to a close-up of car parts being worked on. The camera focuses on the man's hands as he manipulates the car's components, indicating a **shift from narration to demonstration**. The actions are continuous and focused on the same task, so they should be merged into one event. The man's hand movements and the close-up shots are part of the same task, showing the repair process.

- Step 100，reward `1.0000`，repair `False`，answer：**`<events>[[0, 88], [88, 171]]</events>`，完全正确**。  
  CoT：The video starts with the man speaking outdoors, then transitions to a close-up of car parts being worked on. The shots are **distinct in setting and activity**, so each segment is kept separate. The man is speaking and gesturing, then the camera shifts to show mechanical work on the car's undercarriage. These are clear changes in subject and focus, justifying separate events.

分析：

这是“真的学到 CoT 能力”的强 case。模型不仅输出了格式，还学会用“场景和活动变化”作为边界证据。

### 5.2 Logic：从错误选项到稳定事件推理

Run：`logic`  
Task：`event_logic_fill_blank`  
Case：`bFqbB25b1cA`  
GT：`B`  
Reward：`0.0000 -> 1.0000`

逐阶段 CoT：

- Step 0，reward `0.5000`，repair **`True`**，answer：**`B`，答案对但 thought 被截断**。  
  CoT：Looking at the sequence: Step 1 shows a car's infotainment screen with settings related to suspension and tire modes. Step 3 shows a person physically removing a spare tire from the trunk. This implies that the missing step (Step 2) must be a transition or action that logically connects the screen settings to the physical action of removing the spare tire. Option **B ("A person unscrews the retainer holding the spare tire")** is the most logical step that would precede removing the spare tire. The other options either involve unrelated actions (like inflating the tire or tightening lug nuts) or are not contextually connected to

- Step 25，reward `0.0000`，repair `False`，answer：**`D`，错误**。  
  CoT：The sequence starts with a screen showing car settings, then transitions to a person reaching into the trunk and pulling out a spare tire. This suggests that the missing step likely involves the person accessing or interacting with the spare tire in the trunk. Option **D** describes an attempt to pull the tire out without removing the center retainer, which is a plausible action that would naturally precede the action of pulling the tire out. The other options involve actions that are not directly related to the visual sequence or don't fit the context of the steps shown.

- Step 50，reward `1.0000`，repair `False`，answer：**`B`，正确且更短**。  
  CoT：The sequence shows a person reaching into the trunk to pull out a spare tire. This implies that the spare tire is currently **secured in the trunk** and needs to be removed. The most logical step before pulling it out is to **unscrew the retainer** holding it in place. This matches option B.

- Step 75，reward `1.0000`，repair `False`，answer：**`B`，稳定**。  
  CoT：The sequence shows a person interacting with the spare tire in the trunk. Step 1 is about car settings, which is unrelated to the spare tire. Step 3 shows the person pulling out the spare tire, which implies that the missing step (Step 2) must be the action required to remove or access the spare tire. Option **B describes unscrewing the retainer**, which is a logical step before pulling out the spare tire. Options A, C, D, E, and F do not fit the context of the visual sequence.

- Step 100，reward `1.0000`，repair `False`，answer：**`B`，稳定**。  
  CoT：The sequence shows a person interacting with the spare tire in the trunk. Step 1 is about car settings, which is unrelated to the spare tire. Step 3 shows the person pulling out the spare tire, which implies that the missing step (Step 2) must be the action required to remove or access the spare tire. Option **B describes unscrewing the retainer**, which is a logical step before pulling out the spare tire. Options A, C, D, E, and F do not fit the context of the visual sequence.

分析：

这个 case 体现 Logic 的 CoT 不是描述画面，而是在做中间状态推理：从“看到人在拉备胎”反推出缺失步骤应该是“先松开/取出备胎”。这是比纯视觉匹配更强的 process reasoning。

### 5.3 AOT：学到了比较，但预算压力明显

Run：`aot`  
Task：`seg_aot_action_t2v_binary`  
Case：`wSveOVRLkn4`  
GT：`A`  
Reward：`0.0000 -> 1.0000`

逐阶段 CoT：

- Step 0，reward `0.0000`，repair **`True`**，answer：**`B`，错误**。  
  CoT：The paragraph describes a sequence of actions: picking up frozen peas, a woman speaking, taking a garlic clove, and pouring peas into a food processor. In Clip A, the sequence is: hand placing bread slices, woman speaking, hand reaching for fruit stand, hand pouring peas into food processor. This sequence does not match the paragraph's order. The paragraph starts with picking up peas, which is not shown in Clip A. In Clip B, the sequence is: hand picking up peas from countertop, woman speaking, woman taking garlic clove from fruit stand, hand pouring peas into food processor. This matches the paragraph

- Step 25，reward `1.0000`，repair `False`，answer：**`A`，正确**。  
  CoT：The paragraph describes a sequence of actions: picking up peas, speaking, taking garlic, and pouring peas into a food processor. Clip **A** shows the woman speaking, then picking up peas, then taking garlic, and finally pouring peas. This matches the sequence. Clip B shows the woman speaking, then pouring peas, which doesn't match the sequence described in the paragraph. Therefore, Clip A best matches the paragraph.

- Step 50，reward `1.0000`，repair `False`，answer：**`A`，正确且证据更完整**。  
  CoT：The paragraph describes a sequence of actions: picking up peas, a woman speaking, taking a garlic clove, and pouring peas into a food processor. Clip **A shows these exact actions in order**: starting with a hand picking up peas, then the woman speaking, then taking garlic, and finally pouring peas. Clip B, on the other hand, starts with the woman speaking and then shows the peas being poured, which does not match the sequence described in the paragraph. Therefore, Clip A is the correct match.

- Step 75，reward `0.5000`，repair **`True`**，answer：**`A`，答案对但 thought 截断**。  
  CoT：1. The paragraph describes a sequence of actions: first, a hand picks up a bag of frozen peas; then, a woman in a green sweater speaks; next, she takes a garlic clove from a tiered fruit stand; finally, a hand pours green peas into a food processor. 2. In Clip A, the sequence is: a hand picks up a bag of frozen peas, a woman in a green sweater speaks, the woman takes a garlic clove from a tiered fruit stand, and then a hand pours green peas into a food processor. 3. This matches the paragraph exactly. 4. Clip B does not show

- Step 100，reward `1.0000`，repair `False`，answer：**`A`，正确且更规整**。  
  CoT：1. The paragraph describes a sequence of actions: picking up peas, a woman speaking, taking a garlic clove, and pouring peas into a food processor. 2. In Clip A, the sequence is: a hand picks up peas, a woman speaks, she takes garlic, and then peas are poured into a food processor. This matches the paragraph exactly. 3. Clip B does not show these specific actions in this order. 4. Therefore, Clip A is the correct match.

分析：

AOT 的 CoT 是真实的，因为它会逐项比较 paragraph 和 clip 的 action order。但这类任务容易写成长列表，repair ratio 高，说明要么提高 thought budget，要么加入更短的 AOT-specific CoT instruction。

## 6. 风险与限制

1. 缺少严格 no-CoT 对照。
   当前证据证明 CoT teacher run 在训练中变好，但不能单独证明“CoT 比 no-CoT 多带来多少净收益”。

2. CoT format reward 不能等价于 CoT 质量。
   格式正确但语义错误的 case 很常见，尤其在 AOT 和 TG 中。需要 case audit。

3. TG reward 仍偏低。
   TG 的 CoT 很多时候能识别事件，但 precise boundary 仍不稳定。TG 要单独看边界误差、IoU、起止点偏差，而不是只看 thought 是否合理。

4. AOT 的高 repair 是真实风险。
   AOT task-specific repair 在 val100 很高，说明如果继续扩大训练，可能需要更强的 budget control。

## 7. 推荐下一步

1. 做同设置 no-CoT 对照。
   用同模型、同数据、同 step、同 val set 比较 no-CoT vs CoT，才能严谨回答“CoT 训练的净收益”。

2. 给 TG 加专项诊断。
   每个 TG case 输出：GT start/end、pred start/end、start error、end error、IoU、thought 摘要。这样能区分“事件识别错”和“边界错”。

3. AOT 做短 CoT instruction ablation。
   例如要求 thought 最多 3 点：`key visual evidence -> order comparison -> answer`。目标是降低 repair，而不是单纯增加 token budget。

4. Seg L3 需要 granularity 指标。
   统计 predicted segment count vs GT segment count，单独看 under-seg 和 over-seg。当前 L3 的主要问题不是没有 CoT，而是边界粒度偏粗。

## 8. 可复现路径

远端实验根目录：

```text
/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task
```

本地 case 抽取工具：

```text
video_proxy/training/tools/val_cot_change_cases.py
```

典型命令：

```bash
python video_proxy/training/tools/val_cot_change_cases.py \
  --task-filter temporal_grounding \
  --limit 10 \
  --thought-chars 260 \
  logic=/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_logic_100step_cot
```

本报告引用的基础指标来自：

```text
output/doc/qwen3_vl_4b_100step_cot_report.md
```
