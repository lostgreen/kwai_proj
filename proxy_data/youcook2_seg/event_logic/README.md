# Event Logic 数据构造指南

> 基于 YouCook2 标注数据构造的事件逻辑推理 Proxy 任务。

---

## 一、数据源谱系与区分

> ⚠️ **注意**：AoT（`temporal_aot/`）已重构。`build_aot_from_seg.py` 现在直接复用与本任务相同的 L2 分层标注（数据源 B），旧版的独立 VLM captioning 管线（数据源 C）已废弃。

本目录的活跃构建脚本使用 **两套数据源**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     YouCook2 原始视频（共约 2000 个）                  │
└────────────┬────────────────────┬────────────────────┬──────────────┘
             │                    │
     ┌───────▼────────┐  ┌───────▼────────┐
     │  数据源 A       │  │  数据源 B       │
     │ 原始标注        │  │ L2/L3 分层标注  │   ← AoT 新版也用此源
     │ (youcookii_     │  │ (youcook2_seg_  │
     │  annotations_   │  │  annotation/    │
     │  trainval.json) │  │  annotations/)  │
     └───────┬────────┘  └───────┬────────┘
             │                    │
     ┌───────▼────────┐  ┌───────▼────────┐
     │ build_text_    │  │ build_l2_      │   ← 本目录推荐入口
     │ option_proxy   │  │ event_logic    │
     │ .py (旧版)     │  │ .py            │
     └───────┬────────┘  └───────┬────────┘
             │                    │
             ▼                    ▼
     proxy_train_text_   l2_event_logic_
     options.jsonl        *.jsonl
     (add/replace/sort)  (add/replace/sort)
```

### 数据源详情

| 数据源 | 路径 | 标注方式 | 标注粒度 | 备注 |
|--------|------|---------|---------|------|
| **A: 原始标注** | `youcookii_annotations_trainval.json` | YouCook2 官方人工标注 | 视频级，每段一句话 | 旧版管线使用，不推荐 |
| **B: L2 分层标注** | `youcook2_seg_annotation/annotations/*.json` | VLM 自动标注（Qwen2.5-VL-72B） | 三层分层（L1 阶段→L2 活动→L3 动作） | **推荐**；AoT 新版同样使用此源 |

### ⚠️ 关键区分

1. **数据源 A 与 B 的关系**：都基于 YouCook2，但标注体系完全不同。A 是原始一句话描述，B 是三层分层标注（macro phase → event → atomic action）。B 的 `level2.events` 提供了更精细的 `instruction` + `visual_keywords`，质量更可控。

2. **AoT（temporal_aot/）与 Event Logic 的区别**（两者现在共享同一 L2 标注数据源，但任务目标不同）：
   - AoT 关注的是时序**方向感知**（正放 vs 倒放 vs 乱序）
   - Event Logic 关注的是事件**逻辑推理**（下一步/替换/排序）
   - **消融实验中不应将 AoT 数据与 Event Logic 数据混合**，否则无法归因能力提升来自哪个任务

---

## 二、Event Logic 任务定义

### 三种子任务

| 任务 | problem_type | 输入 | 输出 | 考察能力 |
|------|-------------|------|------|---------|
| **Add（预测下一步）** | `add` | N 个连续步骤的视频 + 4 个文本选项 | 选出下一步 | 因果推理 + 事件预测 |
| **Replace（缺失补全）** | `replace` | 含一个 `[MISSING]` 位的步骤序列视频 + 4 个文本选项 | 选出被移除的步骤 | 上下文理解 + 事件关联 |
| **Sort（时序排序）** | `sort` | 打乱顺序的步骤视频 | 输出正确顺序数字序列 | 全局时序理解 |

### 样本格式（EasyR1 JSONL）

```json
{
  "messages": [{"role": "user", "content": "Watch the video...\n<video>\n<video>\n...\nOptions:\nA. Chop the garlic\nB. Boil the water\nC. ...\n\nThink step by step inside <think></think>...\n<answer>B</answer>"}],
  "answer": "B",
  "videos": ["/path/to/event01.mp4", "/path/to/event02.mp4"],
  "data_type": "video",
  "problem_type": "add",
  "metadata": {"video_id": "GLd3aX16zBg", "recipe_type": "113", ...}
}
```

### 奖励函数

| problem_type | Reward | 说明 |
|---|---|---|
| `add` / `replace` | 精确匹配 MCQ | `<answer>X</answer>` 与 GT 字母一致 → 1.0，否则 0.0 |
| `sort` | Jigsaw Displacement | `1 - Σ|pos_pred - pos_gt| / E_max`，连续值 [0,1] |

---

## 三、两条构造管线

### 管线 1：基于原始标注（数据源 A）

```
youcookii_annotations_trainval.json
    │
    │  load_videos(): 按 video_id 遍历
    │  每个 annotation → (segment, sentence, event_id)
    │  拼接 event clip 路径: {root}/{subset}/{recipe}/{video_id}_event{id}_{s}_{e}.mp4
    │
    ▼
build_text_option_proxy.py
    │
    │  Add: 取连续 N 步作为上下文，第 N+1 步为 GT，
    │       负例从同 recipe 的其他视频 sentence 中随机抽取
    │  Replace: 从序列中删除一步，GT=被删步骤的 sentence
    │  Sort: 随机打乱 N 步顺序，GT=原始顺序的数字序列
    │
    ▼
proxy_train_text_options.jsonl  (raw)
    │
    │  filter_bad_videos.py: decord 校验视频可读性
    │
    ▼
proxy_train_text_options_clean.jsonl  (最终)
```

**特点**：
- 依赖预切好的 event clips 目录 (`youcook2_event_clips/`)
- 负例采样基于 `recipe_type` 分组（同类型菜谱内采样，提高难度）
- 标注质量取决于 YouCook2 原始一句话描述（可能过于简短/模糊）

### 管线 2：基于 L2 分层标注（数据源 B）— **推荐**

```
youcook2_seg_annotation/annotations/*.json
    │
    │  load_annotations(): 读取每个 clip 的 level2.events
    │  字段: event_id, start_time, end_time, instruction, visual_keywords
    │
    ▼
build_l2_event_logic.py
    │
    │  同样构造 Add / Replace / Sort，但：
    │  - sentence 来自 L2 的 instruction（更规范、更具体）
    │  - 视频来自 L2 clips 目录（由 prepare_clips.py 截取）
    │  - 可选 AI 因果过滤 (--filter):
    │    VLM 观察帧后判断上下文是否充分、答案是否唯一
    │
    ▼
l2_event_logic_raw.jsonl  (raw)
    │
    │  AI 因果过滤 (可选):
    │  VLM 判断 causal_valid=true + confidence ≥ threshold
    │
    ▼
l2_event_logic_filtered.jsonl  (过滤后)
```

**优势**：
- **标注质量更高**：L2 instruction 由 VLM 72B 模型生成，包含具体动作描述 + 视觉关键词
- **可追溯**：每个 event 有 `parent_phase_id`，可追溯到 L1 宏观阶段
- **可扩展**：新标注的视频只需运行 annotate.py L1→L2→L3 流程
- **AI 质量过滤**：可用 VLM 过滤因果关系不明确的样本
- **与 Hier Seg 任务共享标注**：同一套标注同时支撑 Event Logic 和 Hierarchical Segmentation

---

## 四、底层标注数据格式

### 数据源 A: youcookii_annotations_trainval.json

```json
{
  "database": {
    "GLd3aX16zBg": {
      "subset": "training",
      "recipe_type": "113",
      "annotations": [
        {"id": 0, "segment": [90, 102], "sentence": "spread margarine on two slices of white bread"},
        {"id": 1, "segment": [105, 115], "sentence": "place ham and cheese on one slice"}
      ]
    }
  }
}
```

**字段**：
- `segment`: `[start_sec, end_sec]` 绝对秒数
- `sentence`: 原始一句话描述（英文）
- `recipe_type`: 菜谱类型编号（用于同类型内负例采样）

### 数据源 B: youcook2_seg_annotation/annotations/{clip_key}.json

```json
{
  "clip_key": "GLd3aX16zBg_90_174",
  "video_path": "/path/to/GLd3aX16zBg_90_174.mp4",
  "clip_duration_sec": 84.0,
  "level1": {
    "macro_phases": [
      {"phase_id": 1, "start_time": 0, "end_time": 18,
       "phase_name": "Ingredient Preparation",
       "narrative_summary": "Spreads margarine on bread slices."}
    ]
  },
  "level2": {
    "events": [
      {"event_id": 1, "start_time": 2, "end_time": 16,
       "parent_phase_id": 1,
       "instruction": "Spread margarine evenly on sliced bread",
       "visual_keywords": ["bread", "margarine", "spreading knife"]}
    ]
  },
  "level3": {
    "grounding_results": [
      {"action_id": 1, "start_time": 3, "end_time": 8,
       "parent_event_id": 1,
       "sub_action": "Scoop margarine and spread on first slice",
       "pre_state": "Dry bread slice on cutting board",
       "post_state": "Thin layer of margarine covering the bread surface"}
    ]
  }
}
```

**L2 events 字段**（Event Logic 主要使用）：
- `event_id`: 事件编号
- `start_time` / `end_time`: 秒数（相对 clip 起点）
- `parent_phase_id`: 所属 L1 阶段
- `instruction`: **动作指令**（比数据源 A 的 sentence 更规范）
- `visual_keywords`: 视觉关键词列表

---

## 五、视频切片对照

| 数据源 | 视频切片策略 | 路径模式 | 示例 |
|--------|-------------|---------|------|
| A | 预切好的 event clips | `{root}/{subset}/{recipe}/{vid}_event{id}_{s}_{e}.mp4` | `training/113/GLd3aX16zBg/GLd3aX16zBg_event00_90_102.mp4` |
| B-L2 | 128s 滑窗截取 | `clips/L2/{clip_key}_L2_w{ws}_{we}.mp4` | `clips/L2/GLd3aX16zBg_90_174_L2_w0_128.mp4` |
| B-L3 | event 边界 ±5s 截取 | `clips/L3/{clip_key}_L3_{s}_{e}.mp4` | `clips/L3/GLd3aX16zBg_90_174_L3_2_16.mp4` |
| C | 正放/倒放/乱序 | `{clip_key}.mp4` / `{clip_key}_rev.mp4` / `{clip_key}_shuf.mp4` | `GLd3aX16zBg_event00_90_102_rev.mp4` |

---

## 六、构造命令

### 快速开始（推荐管线 2，基于 L2 标注）

```bash
cd /path/to/VideoProxy/train

# 环境变量
export L2_ANNOTATION_DIR=/path/to/youcook2_seg_annotation/annotations
export L2_CLIPS_DIR=/path/to/youcook2_seg_annotation/clips/L2
export L2_FRAMES_DIR=/path/to/youcook2_seg_annotation/frames
export OUTPUT_DIR=proxy_data/youcook2_seg/event_logic/data

# Step 1: 构造 Event Logic 数据（不过滤）
python proxy_data/youcook2_seg/event_logic/build_l2_event_logic.py \
    --annotation-dir  $L2_ANNOTATION_DIR \
    --clips-dir       $L2_CLIPS_DIR \
    --frames-dir      $L2_FRAMES_DIR \
    --output          $OUTPUT_DIR/l2_event_logic_raw.jsonl \
    --add-per-video 2 --replace-per-video 2 --sort-per-video 1 \
    --min-events 4 --min-context 2 --max-context 4 \
    --seed 42 --shuffle

# Step 2: AI 因果过滤（可选，推荐）
python proxy_data/youcook2_seg/event_logic/build_l2_event_logic.py \
    --annotation-dir  $L2_ANNOTATION_DIR \
    --clips-dir       $L2_CLIPS_DIR \
    --frames-dir      $L2_FRAMES_DIR \
    --output          $OUTPUT_DIR/l2_event_logic_filtered.jsonl \
    --filter \
    --api-base https://api.novita.ai/v3/openai \
    --model qwen/qwen2.5-vl-72b-instruct \
    --confidence-threshold 0.75 \
    --filter-workers 4 \
    --seed 42 --shuffle
```

### 管线 1（基于原始标注，仅作对照）

```bash
# 一键运行
bash proxy_data/youcook2_seg/event_logic/run_build.sh
```

---

## 七、运行流程

### 7.1 Smoketest（冒烟测试）

在正式构造大规模数据前，先用少量样本验证管线是否跑通：

```bash
cd /path/to/EasyR1

# 设置环境变量
export L2_ANNOTATION_DIR=/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations
export L2_CLIPS_DIR=/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/clips/L2
export L2_FRAMES_DIR=/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/frames

# Step 1: 小规模构造（每视频各 1 条，限制总量 20）
python proxy_data/youcook2_seg/event_logic/build_l2_event_logic.py \
    --annotation-dir  $L2_ANNOTATION_DIR \
    --clips-dir       $L2_CLIPS_DIR \
    --frames-dir      $L2_FRAMES_DIR \
    --output          /tmp/el_smoketest_raw.jsonl \
    --add-per-video 1 --replace-per-video 1 --sort-per-video 1 \
    --min-events 4 --min-context 2 --max-context 4 \
    --max-samples 20 \
    --seed 42 --shuffle

# Step 2: 检查输出格式
head -3 /tmp/el_smoketest_raw.jsonl | python3 -m json.tool

# Step 3: 验证视频文件可读
python proxy_data/youcook2_seg/event_logic/filter_bad_videos.py \
    --input-jsonl  /tmp/el_smoketest_raw.jsonl \
    --output-jsonl /tmp/el_smoketest_clean.jsonl

# Step 4: 确认各 problem_type 数量
python3 -c "
import json
from collections import Counter
c = Counter()
with open('/tmp/el_smoketest_raw.jsonl') as f:
    for line in f:
        c[json.loads(line)['problem_type']] += 1
print(dict(c))
"
```

**预期输出**：
- `l2_event_logic_raw.jsonl` 含 add/replace/sort 三种 problem_type
- 每条记录有 `videos` 字段指向真实可读的 .mp4 文件
- `answer` 字段：add/replace 是 A/B/C/D 之一，sort 是数字序列如 "31245"

### 7.2 AI 因果过滤

AI 因果过滤用 VLM 自动判断每条 add/replace 样本的因果质量：
- **上下文是否充分**：视频帧能否提供足够信息区分选项
- **答案是否唯一**：是否只有一个选项是逻辑正确的
- sort 任务不需要过滤（纯物理顺序，无因果歧义）

```bash
# 方式 1: 构造时直接过滤（--filter 开关）
python proxy_data/youcook2_seg/event_logic/build_l2_event_logic.py \
    --annotation-dir  $L2_ANNOTATION_DIR \
    --clips-dir       $L2_CLIPS_DIR \
    --frames-dir      $L2_FRAMES_DIR \
    --output          proxy_data/youcook2_seg/event_logic/data/l2_event_logic_filtered.jsonl \
    --add-per-video 2 --replace-per-video 2 --sort-per-video 1 \
    --filter \
    --api-base https://api.novita.ai/v3/openai \
    --model qwen/qwen2.5-vl-72b-instruct \
    --confidence-threshold 0.75 \
    --filter-workers 4 \
    --seed 42 --shuffle

# 方式 2: 用本地部署的 VLM（更快，无 API 费用）
# 先启动 vLLM 服务:
#   python -m vllm.entrypoints.openai.api_server \
#       --model Qwen/Qwen2.5-VL-72B-Instruct --tp 4 --port 8000
python proxy_data/youcook2_seg/event_logic/build_l2_event_logic.py \
    --annotation-dir  $L2_ANNOTATION_DIR \
    --clips-dir       $L2_CLIPS_DIR \
    --frames-dir      $L2_FRAMES_DIR \
    --output          proxy_data/youcook2_seg/event_logic/data/l2_event_logic_filtered.jsonl \
    --filter \
    --api-base http://localhost:8000/v1 \
    --model Qwen2.5-VL-72B-Instruct \
    --confidence-threshold 0.75 \
    --filter-workers 8 \
    --seed 42 --shuffle
```

**过滤原理**（详见 `prompts.py` 中的 `CAUSALITY_SYSTEM_PROMPT`）：
1. 从 `--frames-dir` 中取每个上下文事件的关键帧（默认 3 帧/事件）
2. 将帧图片 + 选项文本发送给 VLM
3. VLM 返回 `{"causal_valid": bool, "confidence": float, "reason": "..."}`
4. 仅保留 `causal_valid=true` 且 `confidence >= threshold` 的样本

**所需环境变量**：
- `NOVITA_API_KEY` 或 `OPENAI_API_KEY`（使用云端 API 时）
- 本地部署无需 API key

### 7.3 T→V 扩展数据（可选）

如需构造 T→V（文本上下文→视频选项）任务数据：

```bash
# Step 1: 为 L2 event clips 标注 recipe-instruction 风格的 step caption
python proxy_data/youcook2_seg/event_logic/annotate_l2_step_captions.py \
    --from-dataset proxy_data/youcook2_seg/event_logic/data/l2_event_logic_raw.jsonl \
    --output proxy_data/youcook2_seg/event_logic/data/l2_step_captions.jsonl \
    --api-base http://localhost:8000/v1 \
    --model Qwen2.5-VL-72B-Instruct \
    --workers 8

# Step 2: 构造 T→V 训练数据
python proxy_data/youcook2_seg/event_logic/build_l2_event_logic_t2v.py \
    --annotation-dir  $L2_ANNOTATION_DIR \
    --clips-dir       $L2_CLIPS_DIR \
    --frames-dir      $L2_FRAMES_DIR \
    --caption-jsonl   proxy_data/youcook2_seg/event_logic/data/l2_step_captions.jsonl \
    --output          proxy_data/youcook2_seg/event_logic/data/l2_event_logic_t2v.jsonl \
    --seed 42 --shuffle
```

### 7.4 完整标注流程（从新视频开始）

如果需要扩充 Event Logic 训练数据，应**从分层标注入手**（管线 2），不要直接修改 JSONL：

```bash
cd /path/to/EasyR1

# ==== Step 1: 准备原始视频 ====
# 将 YouCook2 原始 mp4 放入指定目录
# 每个视频生成一个 clip_key（格式: {video_id}_{start}_{end}）

# ==== Step 2: 抽帧 ====
python proxy_data/youcook2_seg/youcook2_seg_annotation/extract_frames.py \
    --video-dir /path/to/new_videos \
    --output-dir $L2_FRAMES_DIR \
    --fps 1.0 \
    --workers 4

# ==== Step 3: 逐层 VLM 标注（L1 → L2 → L3）====
# L1: 宏观阶段划分
python proxy_data/youcook2_seg/youcook2_seg_annotation/annotate.py \
    --frames-dir $L2_FRAMES_DIR \
    --output-dir $L2_ANNOTATION_DIR \
    --level 1 \
    --api-base https://api.novita.ai/v3/openai \
    --model qwen/qwen2.5-vl-72b-instruct \
    --workers 2

# L2: 事件细分（在 L1 基础上）
python proxy_data/youcook2_seg/youcook2_seg_annotation/annotate.py \
    --frames-dir $L2_FRAMES_DIR \
    --output-dir $L2_ANNOTATION_DIR \
    --level 2 \
    --api-base https://api.novita.ai/v3/openai \
    --model qwen/qwen2.5-vl-72b-instruct \
    --workers 2

# L3: 原子动作 grounding（在 L2 基础上）
python proxy_data/youcook2_seg/youcook2_seg_annotation/annotate.py \
    --frames-dir $L2_FRAMES_DIR \
    --output-dir $L2_ANNOTATION_DIR \
    --level 3 \
    --api-base https://api.novita.ai/v3/openai \
    --model qwen/qwen2.5-vl-72b-instruct \
    --workers 2

# ==== Step 4: 截取 L2 视频窗口片段 ====
python proxy_data/youcook2_seg/youcook2_seg_annotation/prepare_clips.py \
    --input  $L2_ANNOTATION_DIR/l2_dataset.jsonl \
    --output $L2_ANNOTATION_DIR/l2_dataset_clipped.jsonl \
    --clip-dir $L2_CLIPS_DIR \
    --workers 4

# ==== Step 5: 构造 Event Logic 训练数据 ====
python proxy_data/youcook2_seg/event_logic/build_l2_event_logic.py \
    --annotation-dir $L2_ANNOTATION_DIR \
    --clips-dir      $L2_CLIPS_DIR \
    --frames-dir     $L2_FRAMES_DIR \
    --output         proxy_data/youcook2_seg/event_logic/data/l2_event_logic_raw.jsonl \
    --add-per-video 2 --replace-per-video 2 --sort-per-video 1 \
    --seed 42 --shuffle

# ==== Step 6 (可选): AI 因果过滤 ====
python proxy_data/youcook2_seg/event_logic/build_l2_event_logic.py \
    --annotation-dir $L2_ANNOTATION_DIR \
    --clips-dir      $L2_CLIPS_DIR \
    --frames-dir     $L2_FRAMES_DIR \
    --output         proxy_data/youcook2_seg/event_logic/data/l2_event_logic_filtered.jsonl \
    --filter \
    --api-base https://api.novita.ai/v3/openai \
    --model qwen/qwen2.5-vl-72b-instruct \
    --confidence-threshold 0.75 \
    --filter-workers 4 \
    --seed 42 --shuffle
```

**标注层级关系**：
```
L1 宏观阶段 (macro_phases)
 └─ L2 事件 (events)          ← Event Logic 主要使用
     └─ L3 原子动作 (grounding_results)
```

每层标注依赖上一层的输出，必须**按序执行** L1 → L2 → L3。

### 7.5 消融实验

纯 Event Logic 消融实验脚本位于 `local_scripts/event_logic_ablations/`：

```bash
# 单实验运行
MAX_STEPS=60 bash local_scripts/event_logic_ablations/exp5_all_mixed.sh

# 分 3 台机器批量运行
MAX_STEPS=60 bash local_scripts/event_logic_ablations/run_batch.sh 1  # exp1→exp4
MAX_STEPS=60 bash local_scripts/event_logic_ablations/run_batch.sh 2  # exp2→exp5
MAX_STEPS=60 bash local_scripts/event_logic_ablations/run_batch.sh 3  # exp3→exp6
```

详见 `local_scripts/event_logic_ablations/README.md`。

---

## 八、与消融实验的关系（历史）

### 当前消融实验（aot_ablations）用的数据

| 实验 | 数据 | 数据源 |
|------|------|--------|
| exp1~6 | `aot_v2t` / `aot_t2v` / `aot_4way_v2t` | 数据源 C（AoT） |
| exp7~9 | 交叉实验，混合不同 exp 的过滤后数据 | 数据源 C |

### ⚠️ 问题：不应混合 AoT 与 Event Logic

| | AoT 任务 | Event Logic 任务 |
|---|---|---|
| **考察能力** | 时序方向感知（正/反/乱序） | 事件逻辑推理（预测/补全/排序） |
| **任务形式** | 给定视频/caption，选择方向匹配的选项 | 给定视频序列，选择逻辑正确的文本 |
| **标注来源** | VLM caption + forward/reverse/shuffle 视频 | 分层标注的 instruction |
| **负例构造** | 同 clip 的其他方向 caption | 同菜谱类型的其他步骤 instruction |
| **Reward** | MCQ 精确匹配 | MCQ 精确匹配 / Jigsaw Displacement |

**结论**：如果消融实验想验证 Event Logic 的效果，应**单独**用管线 2 构造的 `l2_event_logic_filtered.jsonl` 训练，不要混合 AoT 数据。两者测量的是不同维度的时序理解能力。

---

## 九、文件清单

```
event_logic/
├── README.md                          ← 本文件
├── README_t2v.md                      ← T→V 扩展方案文档
├── build_text_option_proxy.py         ← 管线 1: 从原始标注构造 add/replace/sort
├── build_l2_event_logic.py            ← 管线 2: 从 L2 标注构造 add/replace/sort (推荐)
├── build_l2_event_logic_t2v.py        ← T→V 扩展: 文本上下文→视频选项
├── annotate_l2_step_captions.py       ← 为 T→V 任务标注 recipe-instruction 风格 caption
├── filter_bad_videos.py               ← 视频可读性校验（含训练采样模拟）
├── merge_datasets.py                  ← 合并多个 JSONL 数据集
├── convert_msswift_to_easyr1.py       ← MS-Swift → EasyR1 格式转换
├── prompts.py                         ← Prompt 模板（MCQ + 因果过滤 + caption 标注）
├── run_build.sh                       ← 一键构建（管线 1 + 管线 2）
└── data/                              ← 输出数据
    ├── proxy_train_text_options.jsonl  ← 管线 1 输出 (V→T)
    ├── proxy_val_text_options.jsonl    ← 验证集
    ├── l2_event_logic_raw.jsonl       ← 管线 2 输出（未过滤）
    ├── l2_event_logic_filtered.jsonl  ← 管线 2 输出（AI 过滤后）
    ├── l2_step_captions.jsonl         ← T→V 用 caption
    ├── mixed_train*.jsonl             ← 混合数据集
    └── ...
```

---

## 十、FAQ

**Q: 为什么要用 L2 标注而不是原始标注？**
- L2 的 `instruction` 比原始 `sentence` 更规范，包含动作主体和宾语
- L2 有 `visual_keywords` 字段，可辅助质量过滤
- L2 与 Hier Seg 任务共享标注，一次标注支撑多种 Proxy 任务

**Q: Event Logic 的 sort 任务和 AoT 的 shuffle 有什么区别？**
- Sort：打乱多个**不同 event** 的顺序，考察跨事件的因果逻辑
- AoT Shuffle：将**同一个 event clip** 等分后打乱帧段，考察帧级时序方向
- 前者是语义级别，后者是感知级别

**Q: 如何判断模型是学会了还是随机猜答？**
- 用 `local_scripts/sample_rollout_analysis.py` 分析 rollout
- 重点关注 `random_index` 指标和 `mid_random` 类别占比
- 高 `random_index` + 高方差 = 随机猜答，能力未习得
