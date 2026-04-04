明白了！之前的提示词确实还是太偏向“物理动作/任务型”视频（cooking/repair/sports），导致电影、剧情、日常 Vlog 等内容因为缺乏“物体状态变化”而被过滤或标注质量差。

要将我们之前总结的**6 类通用视频分层规律**（叙事弧、情绪阶段、镜头语言、微表情等）融入你现有的 `DESIGN.md` Pipeline，关键在于**拓宽 Topology 定义**和**丰富 L3 语义类型**，同时保持代码结构兼容。

以下是**通用化改造后的英文提示词 v2.0**。核心改动点已高亮说明。

---

# 🚀 通用化改造核心策略 (Generalization Strategy)

| 模块 | 原设计 (物理动作偏向) | **新设计 (通用视频兼容)** | 代码兼容性 |
|------|---------------------|------------------------|-----------|
| **Topology** | `sequence` = 物理轨迹 (跑酷/滑雪) | **`sequence` = 叙事/时间流 (电影/Vlog/剧情)** | ✅ 无需改代码，仅改 Prompt 定义 |
| **L1 分割** | 意图/目标转换 (准备/执行/收尾) | **语义阶段 (叙事弧/情绪阶段/场景/任务)** | ✅ 字段名不变，语义拓宽 |
| **L2 分割** | 工作流子目标/完整回合 | **场景单元/互动单元/任务子步** | ✅ 字段名不变，语义拓宽 |
| **L3 微动作** | `state_change` / `repetition_unit` | **新增 `interaction_unit` (对话/社交) / `expression_change` (情绪)** | ⚠️ 需扩展 `micro_type` 枚举判断逻辑 |
| **视觉信号** | 物体状态/工具接触/流体反馈 | **新增 剪辑节奏/光线情绪/视线方向/构图变化** | ✅ 仅 Prompt 文本增强 |

---

# 1️⃣ Merged Prompt: L1+L2+Topology (Generalized v2.0)

```markdown
You are given a {duration}s video clip (timestamps 0 to {duration}) with {n_frames} frames.
Your task has four parts.

## PART 1 — DOMAIN CLASSIFICATION
Classify the video using a two-level taxonomy.
Choose ONE broad category (domain_l1) and ONE fine-grained subcategory (domain_l2)
from the list below:
{domain_taxonomy_str}

## PART 2 — TOPOLOGY CLASSIFICATION (CRITICAL)
Analyze the TEMPORAL STRUCTURE of the visible activity and assign exactly ONE topology_type.

### Topology types (Generalized):
- **procedural**: Task-oriented process with meaningful sub-goals progressing toward an outcome.
  Typical examples: cooking, assembling, repairing, crafting, tutorials.
- **periodic**: Rhythm-oriented repeated cycle of the same motion or operation.
  Typical examples: stretching repetitions, weightlifting, dance routines, factory loops.
- **sequence**: **Narrative or Time-flow oriented continuous traversal, story arc, or episode.**
  Typical examples: **movie scenes, vlogs, travel logs, dog agility runs, parkour, skiing.**
  Key: Look for narrative progression, scene changes, or coherent trajectory.
- **flat**: Single continuous activity with no stable internal hierarchy, or mixed/unclear structure.
  Typical examples: idle talking, background footage, loosely mixed footage, ambient scenes.

### Important topology rules:
- Topology is about temporal structure, NOT about domain label alone.
- **For movies/vlogs: Use `sequence` if there is a clear narrative/scene flow.**
- Repetition alone does NOT imply procedural structure.
- Camera cuts do NOT define topology boundaries (but may signal scene changes).
- If the structure is weak or unclear, choose `flat` rather than inventing hierarchy.

### Also output:
- `topology_confidence`: float from 0.0 to 1.0
- `topology_reason`: one brief sentence explaining the decision

## PART 3 — VIDEO SUMMARY & MACRO PHASES (L1)
Write ONE sentence summarizing the video.
Then segment the video into 1–6 macro phases.

### A macro phase is (Generalized):
- **For procedural/periodic**: A broad stage organized by overall intent or goal shift.
- **For sequence/narrative**: A distinct scene, narrative act, or emotional stage (e.g., Intro → Conflict → Resolution).
- **For flat**: The entire continuous activity (1 phase is valid).

### Rules:
- Skip intros, outros, static non-activity spans, and talking-only spans (unless dialogue is the core content).
- Phases do NOT need to cover the entire video.
- Do NOT split by camera cuts alone (unless it signifies a scene change).
- It is VALID to output only 1 macro phase if the entire video is one continuous routine/scene.

## PART 4 — EVENT DETECTION (L2)
Detect events nested inside each macro phase.
Apply the event definition STRICTLY based on `topology_type`:

### If topology_type = "procedural":
- An event is a multi-step workflow (10–60s) that completes a process sub-goal.
- Group related manipulations together; do NOT fragment into atomic tool motions.
- If a phase consists of a single continuous operation, leave `"events": []`.

### If topology_type = "sequence" (Includes Movies/Vlogs):
- **An event is a complete scene, interaction unit, or narrative beat.**
- Look for changes in: location, characters involved, topic of conversation, or emotional tone.
- Do NOT split by every camera cut; group shots belonging to the same scene/interaction.

### If topology_type = "periodic":
- Events are OPTIONAL. You may leave `"events": []` for a phase.
- Do NOT create one event per repetition.
- If you output an event, it should match the whole phase as a container for micro annotation.

### If topology_type = "flat":
- Output `"events": []`. Do NOT invent L2 structure.

### General L2 rules:
- Events must not overlap.
- Use absolute integer seconds (relative to full video timeline).
- It is VALID for a phase to contain zero events.
- Do not force extra events to make the hierarchy deeper.

## OUTPUT FORMAT (JSON)
{{
  "domain_l1": "string",
  "domain_l2": "string",
  "topology_type": "procedural | periodic | sequence | flat",
  "topology_confidence": 0.0-1.0,
  "topology_reason": "string",
  "l2_mode": "workflow | episode | interaction | optional | skip",
  "summary": "string",
  "global_phase_criterion": "string (explain L1 splitting logic, content-agnostic)",
  "macro_phases": [
    {{
      "phase_id": 1,
      "start_time": int,
      "end_time": int,
      "phase_name": "string",
      "narrative_summary": "string",
      "event_split_criterion": "string (explain L2 splitting logic for this phase, content-agnostic)",
      "events": [
        {{
          "event_id": 1,
          "start_time": int,
          "end_time": int,
          "instruction": "string",
          "visual_keywords": ["string", "..."]
        }}
      ]
    }}
  ]
}}

## VISUAL SIGNAL REFERENCE (Generalized)
Use these signals to justify boundaries — be specific and visually verifiable:
- **Scene/Space**: Background/layout/location change, character entry/exit.
- **Subject Behavior**: Pose transition, gaze direction, speed change, interaction start/end.
- **Object State**: Appearance/texture/color/position/quantity change (for procedural).
- **Narrative/Emotion**: **Shift in emotional tone (tense→relaxed), topic change, conflict resolution.**
- **Camera/Editing**: **Significant rhythm change, montage sequence, focus shift, cut to close-up.**
- **Lighting/Mood**: **Lighting shift (dark→bright), color grade change signaling mood.**

## QUALITY CHECKLIST (self-verify before output)
- [ ] Each L1 phase represents a distinct semantic stage (task/narrative/emotional)?
- [ ] Each L2 event (if any) completes a verifiable unit (sub-goal/scene/interaction)?
- [ ] Boundary triggers are specific and reproducible by another annotator?
- [ ] Criterion fields describe splitting LOGIC, not video content?
- [ ] topology_type matches temporal structure (including narrative flow)?
```

---

# 2️⃣ L3 Prompt: Topology-Aware Micro Grounding (Generalized v2.0)

```markdown
You are a temporal grounding model. You are viewing frames from a clip 
({clip_start}s to {clip_end}s).

The input query is: "{action_query}"
The topology_type of the source video is: "{topology_type}"

Your task is to pinpoint every atomic micro-action in this clip.

## IMPORTANT
- If topology_type is "flat", this prompt should not be used (L3 skipped).
- Use absolute integer seconds from the FULL VIDEO timeline (not clip-relative).
- Typical micro-action duration: 2–6 seconds, but adapt based on content type.

## LEVEL 3 DEFINITIONS (Switch by topology_type & content)

### If topology_type = "procedural":
**micro_type = "state_change"**
Find brief atomic actions where an OBJECT undergoes a clear visible PHYSICAL change.
✅ Valid: Cutting, pouring, attaching, spreading, separating.
❌ Ignore: Reaching, idle pauses, pure hand repositioning.
📐 Boundary: start = contact/onset, end = new state established.

### If topology_type = "periodic":
**micro_type = "repetition_unit"**
Find each individual completed repetition, cycle, strike, or stretch.
✅ Valid: One push-up, one jump rope cycle, one stretching rep.
📐 Boundary: start = initiation, end = return to starting position.

### If topology_type = "sequence" (Movies/Vlogs/Narrative):
**micro_type = "interaction_unit" OR "expression_change"**
- **interaction_unit**: A complete social/physical interaction beat.
  ✅ Valid: Handshake start→end, object handover, dialogue turn (visual cue), gesture completion.
- **expression_change**: A distinct shift in facial emotion or focus.
  ✅ Valid: Smile→Serious, Look away→Eye contact, Surprise reaction.
📐 Boundary: start = onset of interaction/expression, end = completion/return to neutral.

## GENERAL RULES
- Allow gaps between micro-actions (do not force full coverage).
- Merge uninterrupted motion belonging to the same single unit.
- For each micro-action, provide EXPLICIT visual state descriptions:
  - `pre_state`: The visual state BEFORE (specific & observable).
  - `post_state`: The visual state AFTER (specific & observable).

## OUTPUT FORMAT (JSON)
{{
  "micro_type": "state_change | repetition_unit | interaction_unit | expression_change",
  "micro_split_criterion": "string (explain L3 splitting logic, content-agnostic)",
  "grounding_results": [
    {{
      "action_id": 1,
      "start_time": int,
      "end_time": int,
      "sub_action": "brief description of the specific interaction/change",
      "pre_state": "explicit visual state BEFORE",
      "post_state": "explicit visual state AFTER",
      "parent_event_id": int (optional),
      "parent_phase_id": int (optional)
    }}
  ]
}}

## VISUAL SIGNAL REFERENCE (for micro boundaries)
- **Object State**: Texture/color/shape/position/quantity visibly altered.
- **Contact Event**: Tool/hand first touches object, or releases after interaction.
- **Social Cue**: **Hand extension, eye contact established/broken, head nod.**
- **Emotional Cue**: **Eyebrow raise,嘴角上扬 (mouth corner lift), tension release.**
- **Camera/Editing**: **Cut to reaction shot, zoom in on face.**

## QUALITY CHECKLIST
- [ ] Each micro-action represents exactly ONE visible unit (state/rep/interaction/expression)?
- [ ] pre_state/post_state are specific enough to be verified by looking at frames?
- [ ] Boundaries align with visible onset/completion, not arbitrary cuts?
- [ ] Criterion describes splitting LOGIC, not specific objects/actions?
- [ ] No hallucinated actions beyond what frames show?
```

---

# 🔧 代码适配建议 (Minimal Code Changes)

为了支持通用化，你的 Python 代码只需做**极小改动**（主要是枚举扩展）：

### 1. `prompts.py`
- 替换上述两套 Prompt 模板。
- 确保 `{domain_taxonomy_str}` 包含足够的非物理类标签（如 `entertainment`, `vlog`, `movie_scene`）。

### 2. `annotate.py` (L3 路由逻辑)
原逻辑只判断 `procedural` 和 `periodic`，需扩展支持 `sequence`：

```python
# 原逻辑
if topology_type == "periodic":
    l3_mode = "repetition_unit"
elif topology_type == "procedural":
    l3_mode = "state_change"
else:
    l3_mode = "skip"

# 新逻辑 (支持通用)
if topology_type == "periodic":
    l3_mode = "repetition_unit"
elif topology_type == "procedural":
    l3_mode = "state_change"
elif topology_type == "sequence":
    # 让 VLM 在 L3 Prompt 中自行判断是 interaction 还是 expression
    l3_mode = "interaction_unit"  # 默认值，Prompt 中允许覆盖
else:
    l3_mode = "skip"
```

### 3. `seg_source.py` (训练数据筛选)
原逻辑可能过滤了 `micro_type != state_change` 的数据。需放宽过滤条件：

```python
# 原逻辑 (可能)
if ann['level3']['micro_type'] not in ['state_change', 'repetition_unit']:
    return False

# 新逻辑 (通用)
valid_micro_types = ['state_change', 'repetition_unit', 'interaction_unit', 'expression_change']
if ann['level3']['micro_type'] not in valid_micro_types:
    return False
```

### 4. `rewrite_criteria_hints.py`
确保改写逻辑能处理“叙事/情绪”类的 criterion，不要强行改写成“任务”口吻。
- 原：`"Segmented by removing wires..."` → `"Segmented by distinct sequential sub-tasks..."`
- 新：`"Segmented by scene change from kitchen to living room..."` → `"Segmented by distinct semantic units based on location or interaction context."`

---

# 📊 预期效果对比

| 视频类型 | 原 Pipeline 表现 | **新 Prompt 表现** |
|---------|----------------|------------------|
| **做饭/维修** | ✅ 优秀 (state_change) | ✅ 保持优秀 (state_change) |
| **健身/拉伸** | ✅ 优秀 (repetition_unit) | ✅ 保持优秀 (repetition_unit) |
| **电影片段** | ❌ 差 (无物体状态变化，被过滤) | ✅ **好 (sequence + interaction_unit)** |
| **日常 Vlog** | ❌ 差 (被归为 flat，无 L3) | ✅ **好 (sequence + interaction/expression)** |
| **真人秀** | ❌ 差 (情绪变化无法标注) | ✅ **好 (sequence + expression_change)** |

---

# 🚀 下一步执行建议

1.  **替换 Prompt**：将上述两套英文 Prompt 更新到你的 `prompts.py`。
2.  **小批量测试**：选 5 条非物理视频（电影/Vlog/真人秀）+ 5 条物理视频，跑 `merged` + `L3`。
3.  **检查 L3 类型分布**：确认 `sequence` 类视频是否成功输出了 `interaction_unit` 或 `expression_change`。
4.  **放宽筛选器**：修改 `seg_source.py` 确保新 `micro_type` 能进入训练集。

这样既保留了你现有的 Pipeline 架构（不用重写代码），又通过 Prompt 语义扩展实现了“通用分层”的目标。是否需要我帮你写一个具体的 `seg_source.py` 修改片段？