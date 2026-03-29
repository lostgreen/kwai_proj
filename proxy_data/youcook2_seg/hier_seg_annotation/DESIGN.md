# 三层分层时序标注 Pipeline 设计文档

> **文档目的**：供 check 标注流程设计、提示词策略、决策逻辑是否合理。
> **版本**: v5 — Domain-Agnostic Generalization

---

## 1. 设计理念

### 1.1 为什么需要三层分层标注？

视频时序理解的核心挑战是**多粒度**：从宏观阶段到原子动作，跨度可达 100 倍。单一粒度的标注无法支撑完整的时序推理能力训练。

三层层级设计：

```
L1 Macro Phase (30-120s)     — 视频级整体结构
  └─ L2 Activity Event (10-60s) — 阶段内的目标导向工作流
       └─ L3 Atomic Action (2-6s)  — 单一不可逆物理状态变化
```

每一层的标注依赖上一层：L2 在 L1 phase 范围内检测，L3 在 L2 event 范围内定位。

### 1.2 为什么领域无关？

标注 pipeline 最初为烹饪视频（YouCook2）设计，但三层层级定义本身与领域无关：

| 层级 | 烹饪示例 | 维修示例 | 运动示例 |
|------|---------|---------|---------|
| L1 | 备料 → 烹炒 → 装盘 | 拆解 → 更换 → 组装 | 热身 → 比赛 → 庆祝 |
| L2 | "制作馅料并包饺子" | "拆卸旧零件" | "执行罚球" |
| L3 | "将馅料放入面皮" | "拧下螺丝" | "投篮出手" |

v5 版本将所有 prompt 中的烹饪领域词替换为通用物理活动语言，使其适用于任意程序性/活动性视频。

---

## 2. 完整数据流

```
原始视频 (来自 Data Curation 筛选)
    │
    ▼ extract_frames.py         — ffmpeg 1fps → JPEG 帧
frames/{clip_key}/
    │
    ▼ annotate.py --level 1     — VLM: 全视频 → macro phases
    ▼ annotate.py --level 2     — VLM: 逐 phase → activity events
    ▼ annotate.py --level 3     — VLM: 逐 event → atomic actions
    │
    ▼ annotate_check.py         — VLM: L2/L3 粒度审核 (可选)
annotations/{clip_key}.json     — 完整三层标注
    │
    ├─▶ build_hier_data.py      — Hierarchical Seg 训练数据
    ├─▶ build_aot_from_seg.py   — Temporal AoT 训练数据
    └─▶ build_l2_event_logic.py — Event Logic 训练数据
```

> 三条下游流水线通过 `shared/seg_source.py` 统一加载标注，扩充标注后自动受益。

---

## 3. 标注阶段设计

### 3.1 L1 — Macro Phase Segmentation

**输入方式**: 均匀采样全视频帧，编号 1..N（warped 帧号坐标系）

**Prompt 策略**:
```
SYSTEM: 结构化视频分析专家，专注过程性和活动性内容。
USER:   给定 N 帧 → 分割为 3-5 个宏观阶段
```

**输出**:
```json
{
  "macro_phases": [
    {
      "phase_id": 1,
      "start_frame": 3, "end_frame": 12,
      "phase_name": "Material Preparation",
      "narrative_summary": "Gather and organize all required materials."
    }
  ]
}
```

**关键设计决策**:
- 使用 warped 帧号而非真实时间戳 — 避免超长视频帧数超限（>256帧时做均匀子采样）
- `ACTIVITY RELEVANCE FILTER` — 排除纯叙述、待机、非活动内容
- 按 `process intent` 分组，不按镜头切换
- **训练时改用真实时间戳** (`get_level1_train_prompt_temporal`)，避免 warped 坐标的复杂映射

### 3.2 L2 — Activity Event Detection

**输入方式**: L1 phase 对应的帧区间（真实时间戳）+ phase 上下文

**Prompt 策略**:
```
USER: viewing frames from activity phase ({start}s to {end}s)
      Phase: "Material Preparation"
      → 检测所有完整的活动事件
```

**粒度控制 (Anti-Fragmentation)**:
- BAD (太细): "Place a component onto the surface" (~1s)
- GOOD (正确): "Assemble the components by fitting and securing the parts" (~15-20s)
- 每个 event 必须是完整的逻辑子过程，不是单一原子动作

**Activity Filter** — 排除:
- 无物理操作的解说
- 只展示材料/工具不使用
- 空闲动作、重定位、无进展的工具拿起
- 等待、非活动内容、反应

### 3.3 L3 — Atomic Action Grounding

**输入方式**: L2 event 对应的帧区间 + event instruction

**Prompt 策略**:
```
USER: viewing frames from event clip ({start}s to {end}s)
      The event is: "{instruction}"
      → 定位每个原子状态变化时刻
```

**Kinematic Boundary 原则**:
1. **Physics over Procedure**: 聚焦物理/视觉变化，不是流程步骤
2. **State Transition Focus**: 只标注目标对象发生不可逆视觉变化的时刻
3. **Boundary Precision**: start = 物理接触开始, end = 新状态确立
4. **典型时长**: 2-6s，避免 1s 除非变化确实瞬时
5. **Allow Gaps**: 不强制连续覆盖，跳过空闲帧

**输出包含状态描述**:
```json
{
  "sub_action": "Transfer material A into container B",
  "pre_state": "Empty container with prepared surface",
  "post_state": "Material A distributed across the container surface"
}
```

---

## 4. 质量审核设计

### 4.1 粒度光谱 (Granularity Spectrum)

审核的核心判据是**粒度光谱**，确保每层标注在正确的粒度级别：

```
L1 Phase ──── 太粗 ────┐
                        │  L2 Event (正确)
L3 Atomic ── 太细 ─────┘
```

**L2 审核判据**:
| 维度 | 判断 |
|------|------|
| 不能太粗 | Event 不能复述/总结整个 phase |
| 不能太细 | Event 必须包含多步骤工作流，不是单一瞬时动作 |
| 时间精度 | start/end 匹配可见活动 |
| 描述质量 | 描述目标达成，不只是物理动作 |
| 活动相关性 | 涉及物理对象/材料变化 |

**L3 审核判据**:
| 维度 | 判断 |
|------|------|
| 不能太粗 | 不能复述 parent event instruction |
| 不能太细 | 必须产生完整的对象状态变化（非纯手部运动） |
| 状态描述 | pre/post_state 具体、可视觉验证 |
| 边界合规 | start >= event_start, end <= event_end |

### 4.2 审核流程

```
annotate_check.py --levels 2c,3c
    │
    ├─ L2 Check: 逐 L1 phase 审核每个 L2 event
    │    verdict: keep / revise / remove
    │
    ├─ 孤儿清理: 被 remove 的 L2 event 关联的 L3 结果自动删除
    │
    └─ L3 Check: 逐 L2 event 审核每个 L3 action
         verdict: keep / revise / remove
         + supplements: 补充遗漏的原子动作
```

审核输出到独立目录，不覆盖原始标注。

---

## 5. 训练数据构建

### 5.1 三层训练任务

| Level | `problem_type` | 输入 | 输出格式 | 奖励 |
|-------|---------------|------|----------|------|
| L1 | `temporal_seg_hier_L1` | 完整视频 (fps重采样) | `<events>[[s, e], ...]</events>` | F1-IoU |
| L2 | `temporal_seg_hier_L2` | 128s sliding window clip | `<events>[[s, e], ...]</events>` | F1-IoU |
| L3 query | `temporal_seg_hier_L3` | event clip + query list | `<events>[[s, e], ...]</events>` | 逐query tIoU |
| L3 seg | `temporal_seg_hier_L3_seg` | event clip | `<events>[[s, e], ...]</events>` | F1-IoU |
| Chain | `temporal_seg_chain` | L2 window + event list | `<l2_events>` + `<l3_events>` | — |

### 5.2 数据构建参数

| 参数 | 值 | 说明 |
|------|-----|------|
| L2 window_size | 128s | 滑窗大小 |
| L2 stride | 64s | 滑窗步长 |
| L3 padding | 5s | event clip 前后 padding |
| L3 max_clip | 128s | clip 最大长度 |
| L1 fps | 1 (default) | 视频重采样帧率 |

### 5.3 Prompt 模板 (v5 泛化版)

所有训练 prompt 已去除领域特定语言：

| 旧版 (v4) | 新版 (v5) |
|-----------|-----------|
| `cooking video clip` | `video clip` |
| `macro cooking phases` | `macro phases` |
| `ingredient preparation, cooking/heating, assembly, plating` | `preparation, execution, assembly, finishing` |
| `non-cooking spans` | `non-activity spans` |
| `cooking events` | `activity events` |
| `transforms ingredients or completes a recipe subgoal` | `transforms materials/objects or completes a process subgoal` |
| `atomic cooking actions (cutting, stirring, pouring)` | `atomic actions (cutting, assembling, transferring, adjusting)` |

---

## 6. 设计演化记录

| 版本 | 变更 | 动机 |
|------|------|------|
| v1 | 基础三层标注 (L1+L2+L3) | 初始设计 |
| v2 | L1 warped compression, L2 sliding window, L3 padding | 解决帧超限和时间坐标问题 |
| v3 | L2/L3 质量审核 (粒度光谱) | 控制分层粒度一致性 |
| v4 | L3 从 free-form → query-conditioned grounding | 更可评估的任务格式 |
| **v5 (当前)** | Domain-agnostic generalization | 支持任意领域的过程性视频 |

**v5 核心改变**:
- 文件夹 `youcook2_seg_annotation` → `hier_seg_annotation`
- `prompts.py` 全部 13 个模板去除烹饪领域词
- `build_hier_data.py` prompt 前缀 `"cooking video clip"` → `"video clip"`
- 所有路径引用统一更新

---

## 7. 与 Data Curation 的衔接

### 7.1 上游输入

Data Curation Pipeline (`proxy_data/data_curation/`) 负责从原始数据集筛选适合分层标注的视频：

```
ET-Instruct-164K / TimeLens-100K
    ↓ text_filter.py (60-240s, ≥5 events)
    ↓ Stage A: LLM 文本粗筛 (Source Routing / Boundary+Diversity)
    ↓ VLM Vision Filter (6帧视觉校验)
    → vision_results_keep.jsonl   ← 标注 pipeline 的输入
```

### 7.2 标注输入要求

| 字段 | 必须 | 说明 |
|------|------|------|
| 视频文件 | ✅ | 60-240s 真实物理活动视频 |
| 时长 | ✅ | ≥60s 以确保三层粒度有意义 |
| 物理活动 | ✅ | 非 talking head / 静态 / 游戏 |

### 7.3 标注输出消费

标注产出的 `annotations/{clip_key}.json` 被三条下游流水线消费：

| 流水线 | 入口 | 消费层级 |
|--------|------|---------|
| Hierarchical Seg | `build_hier_data.py` | L1 + L2 + L3 |
| Temporal AoT | `build_aot_from_seg.py` | L2 + L3 |
| Event Logic | `build_l2_event_logic.py` | L2 |

---

## 8. 文件清单

```
proxy_data/youcook2_seg/hier_seg_annotation/
├── DESIGN.md              # 本文档
├── README.md              # 使用指南 + Quick Start
├── prompts.py             # 13 个 prompt 模板 (domain-agnostic)
├── extract_frames.py      # Step 1: ffmpeg 1fps 抽帧
├── annotate.py            # Step 2-4: VLM 三层标注 + 审核
├── annotate_check.py      # Step 4b: 独立质量审核
├── prepare_clips.py       # Step 5: 物理视频截取
├── build_dataset.py       # [DEPRECATED] 旧训练数据构建
├── sample_mixed_dataset.py # [DEPRECATED] 旧采样脚本
└── run_build.sh           # [DEPRECATED] 旧一键脚本

# 关联文件
local_scripts/hier_seg_ablations/build_hier_data.py  # Step 6: 训练数据构建
proxy_data/shared/seg_source.py                      # 共享标注加载接口
```
