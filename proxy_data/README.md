# Proxy Data 构造工具集

基于 YouCook2 数据集构造的多种 Proxy 训练任务，用于视频时序理解和推理能力训练。
所有任务输出统一为 **EasyR1 JSONL 格式**，可直接用于模型训练或混合使用。

---

## 一、整体架构

### 目录组织（重构后）

```
proxy_data/
├── shared/                    ← 三条流水线统一接口（seg_source.py）
├── youcook2_seg/              ← ★ 共享 YouCook2 层次分割标注的三条流水线
│   ├── youcook2_seg_annotation/  ← 标注数据源 + 标注工具
│   ├── temporal_aot/             ← AoT 任务（从 seg 标注构建）
│   └── event_logic/              ← 事件逻辑任务（从 L2 标注构建）
└── temporal_grounding/        ← 独立数据源（TimeRFT，与 seg 无关）
```

### 数据来源（单一标注源，三条流水线共享）

```
youcook2_seg/youcook2_seg_annotation/annotations/*.json
    ↕ 三条流水线共享同一标注源，通过 shared/seg_source.py 统一加载
```

每个 `{clip_key}.json` 包含三层层级标注：
- **L1** `level1.macro_phases`：宏观烹饪阶段（3–5 个）
- **L2** `level2.events`：烹饪事件（128s 滑窗内，2–10 个）
- **L3** `level3.grounding_results`：原子动作（每个 event 内，3–6 个）

### 三条 Proxy 数据流水线

```
youcook2_seg/youcook2_seg_annotation/annotations/*.json (共享标注源)
       │
       ├─► [hier_seg]    build_hier_data.py      → L1/L2/L3/L3_seg 时序分割任务
       │
       ├─► [temporal_aot] build_aot_from_seg.py  → action/event 顺序判断 MCQ 任务
       │
       └─► [event_logic]  build_l2_event_logic.py → add/replace/sort 事件逻辑任务
```

### 共享接口模块

**`shared/seg_source.py`** — 三条流水线的统一底层接口：

| 功能 | 接口 |
|------|------|
| 标注加载 | `load_annotations(ann_dir, complete_only=False)` |
| L2 滑窗生成 | `generate_sliding_windows(total_duration, ...)` |
| L3 clip 窗口计算 | `compute_l3_clip(ev_start, ev_end, clip_duration, ...)` |
| L1 clip 路径 | `get_l1_clip_path(clip_key, clip_dir_l1, fps=1)` |
| L2 clip 路径 | `get_l2_clip_path(clip_key, ws, we, clip_dir_l2)` |
| L3 clip 路径 | `get_l3_clip_path(clip_key, event_id, cs, ce, clip_dir_l3)` |
| 常量 | `L2_WINDOW_SIZE=128`, `L2_STRIDE=64`, `L3_PADDING=5`, `L3_MAX_CLIP_SEC=128` |

---

## 二、Proxy 任务设计总览

| Proxy 类型 | 目录 | 任务描述 | problem_type | 状态 |
|-----------|------|---------|-------------|------|
| **Hierarchical Seg** | `youcook2_seg/youcook2_seg_annotation/` + `local_scripts/hier_seg_ablations/` | 三层分层视频分割（阶段/活动/动作） | `temporal_seg_hier_L1/L2/L3/L3_seg` | ✅ 可用 |
| **Temporal AoT (from seg)** | `youcook2_seg/temporal_aot/` | 从 seg 标注构建 action/event 顺序判断 MCQ | `seg_aot_action_v2t/t2v`, `seg_aot_event_v2t/t2v` | ✅ 可用 |
| **Event Logic** | `youcook2_seg/event_logic/` | 基于 L2 事件的 add/replace/sort 推理 | `add` / `replace` / `sort` | ✅ 可用 |
| **Temporal Grounding** | `temporal_grounding/` | 独立来源（TimeRFT），时序定位 | `temporal_grounding` | ✅ 可用 |

---

## 三、各 Proxy 任务详细设计

### 3.1 Hierarchical Segmentation（三层分层时序分割）

**构建入口**：`local_scripts/hier_seg_ablations/build_hier_data.py`

**标注数据源**：`youcook2_seg/youcook2_seg_annotation/annotations/`

**用法**：
```bash
python build_hier_data.py \
    --annotation-dir /path/to/youcook2_seg/youcook2_seg_annotation/annotations \
    --clip-dir-l1 /path/to/clips/L1 \   # 可选，L1 fps-重采样 clips
    --clip-dir-l2 /path/to/clips/L2 \   # 可选，L2 窗口 clips
    --clip-dir-l3 /path/to/clips/L3 \   # 可选，L3 event clips
    --output-dir ./data/hier_seg \
    --levels L1 L2 L3 L3_seg \
    --l1-fps 1                           # L1 视频帧率，默认 1fps
```

#### L1 — 宏观阶段分割（Macro Phase Segmentation）

| 项目 | 说明 |
|-----|------|
| **粒度** | 阶段级（完整原始视频） |
| **输入视频** | 原始视频按 `--l1-fps`（默认 1fps）重采样的 MP4，由 `prepare_clips.py` 生成 |
| **视频命名** | `{clip_key}_L1_{fps}fps.mp4` |
| **Prompt** | 给出真实时长 0~{duration}s，要求将视频分割为 3–5 个高层语义阶段 |
| **输出格式** | `<events>[[0, 85], [90, 170], [180, 240]]</events>`（真实秒数） |
| **时间坐标系** | 真实秒数（0-based） |
| **problem_type** | `temporal_seg_hier_L1` |

#### L2 — 滑窗事件检测（Sliding-Window Event Detection）

| 项目 | 说明 |
|-----|------|
| **粒度** | 活动级（128s 窗口） |
| **输入视频** | `{clip_key}_L2_w{ws}_{we}.mp4`（ffmpeg `-ss ws -t 128` 截取） |
| **Prompt** | 给出片段时长，要求检测所有完整烹饪事件及时间边界 |
| **输出格式** | `<events>[[5, 42], [55, 90]]</events>`（窗口内 0-based 秒数） |
| **滑窗参数** | `window=128s, stride=64s, min_events=2` |
| **problem_type** | `temporal_seg_hier_L2` |

#### L3 — 查询驱动原子动作定位（Query-conditioned Atomic Grounding）

| 项目 | 说明 |
|-----|------|
| **粒度** | 动作级（event clip 内） |
| **输入视频** | `{clip_key}_L3_ev{id}_{cs}_{ce}.mp4`（event 边界 ±5s padding，max 128s） |
| **Prompt** | 给出编号 action 列表（可打乱），要求按序输出每个 action 的时间区间 |
| **输出格式** | `<events>[[3, 7], [0, 2], [10, 14]]</events>`（按 query 顺序，clip 内 0-based） |
| **problem_type** | `temporal_seg_hier_L3` |

#### L3_seg — 自由原子动作分割（Free Atomic Segmentation）

| 项目 | 说明 |
|-----|------|
| **输入视频** | 同 L3（相同 clip 文件，可复用） |
| **Prompt** | 无 query，直接检测所有原子动作时间段 |
| **输出格式** | `<events>[[3, 7], [10, 14], [16, 22]]</events>` |
| **problem_type** | `temporal_seg_hier_L3_seg` |

**clip 准备**：
```bash
# 准备 L1 fps-重采样 clips（新流程）
python prepare_clips.py --input train.jsonl --output train_l1.jsonl \
    --clip-dir /clips/L1 --l1-fps 1

# 准备 L2 / L3 clips（ffmpeg 截取）
python prepare_clips.py --input train.jsonl --output train_l2.jsonl \
    --clip-dir /clips/L2
```

---

### 3.2 Temporal Arrow of Time（时序方向判断，从 seg 标注构建）

**构建入口**：`youcook2_seg/temporal_aot/build_aot_from_seg.py`

直接复用 seg annotation 的 L2 events 和 L3 grounding_results，无需独立 VLM captioning。

#### 四种任务类型（2×2 factorial）

| 任务 | problem_type | 粒度 | 形式 |
|------|-------------|------|------|
| **action V2T** | `seg_aot_action_v2t` | L3 (action) | 给 L3 event clip，判断哪个动作列表顺序正确（A/B 二选一） |
| **action T2V** | `seg_aot_action_t2v` | L3 (action) | 给 forward 动作列表，从两个 L3 clip 中选匹配的（A/B 二选一） |
| **event V2T** | `seg_aot_event_v2t` | L2 (event) | 给 L2 window clip，判断哪个事件列表顺序正确（A/B/C 三选一） |
| **event T2V** | `seg_aot_event_t2v` | L2 (event) | 给 forward 事件列表，从三个 L2 clip 中选匹配的（A/B/C 三选一） |

**用法**：
```bash
python proxy_data/youcook2_seg/temporal_aot/build_aot_from_seg.py \
    --annotation-dir /path/to/youcook2_seg/youcook2_seg_annotation/annotations \
    --clip-dir-l2 /path/to/clips/L2 \
    --clip-dir-l3 /path/to/clips/L3 \
    --output-dir /path/to/output \
    --tasks action_v2t action_t2v event_v2t event_t2v \
    --complete-only
```

---

### 3.3 Event Logic（L2 事件逻辑推理）

**构建入口**：`youcook2_seg/event_logic/build_l2_event_logic.py`

基于 L2 事件标注构造 add / replace / sort 任务，使用 L2 window clips 作为视频输入。

| 任务类型 | 设计 |
|---------|------|
| **Add** | 给定连续 N 步视频，选择下一步对应的事件描述（文本选项） |
| **Replace** | 给定含缺失步的视频序列，选择正确填补的描述 |
| **Sort** | 给定打乱顺序的视频片段，输出正确时序排列 |

**用法**：
```bash
python proxy_data/youcook2_seg/event_logic/build_l2_event_logic.py \
    --annotation-dir proxy_data/youcook2_seg/youcook2_seg_annotation/annotations \
    --clips-dir /path/to/clips/L2 \
    --output proxy_data/youcook2_seg/event_logic/data/l2_event_logic.jsonl
```

---

## 四、视频 Clip 命名规范（统一，由 shared/seg_source.py 维护）

| 层级 | 命名 | 说明 |
|------|------|------|
| L1 | `{clip_key}_L1_{fps}fps.mp4` | ffmpeg `-vf fps=N` 全视频重采样 |
| L2 | `{clip_key}_L2_w{ws}_{we}.mp4` | ffmpeg `-ss ws -t duration` 窗口截取 |
| L3 | `{clip_key}_L3_ev{event_id}_{cs}_{ce}.mp4` | ffmpeg event clip（±5s padding） |

---

## 五、统一数据格式（EasyR1 JSONL）

```json
{
  "messages": [{"role": "user", "content": "...（含 <video> 占位符）"}],
  "prompt": "...（同 messages[0].content）",
  "answer": "A  或  <events>[[s,e],...]</events>  或  12345",
  "videos": ["path/to/clip.mp4"],
  "data_type": "video",
  "problem_type": "temporal_seg_hier_L1 | L2 | L3 | L3_seg | seg_aot_* | add | replace | sort",
  "metadata": {
    "clip_key": "...",
    "clip_duration_sec": 84,
    "level": 1,
    "l1_fps": 1,
    "source_video_path": "/path/to/original.mp4",
    ...
  }
}
```

**L1 metadata 特殊字段**：
- `l1_fps`：重采样帧率（默认 1）
- `source_video_path`：原始视频路径，供 `prepare_clips.py` 生成 fps clip 时使用

---

## 六、目录结构

```
proxy_data/
├── shared/                              # 三条流水线统一接口
│   ├── __init__.py
│   └── seg_source.py                   #   常量 + 标注加载 + clip路径命名 + 几何计算
│
├── youcook2_seg/                        # ★ 共享 YouCook2 层次分割标注的三条流水线
│   │
│   ├── youcook2_seg_annotation/         # 三层分层标注数据源
│   │   ├── annotations/                #   ★ 核心：所有 *.json 标注文件（三条流水线共享）
│   │   ├── clips/                      #   物理 clip 文件（由 prepare_clips.py 生成）
│   │   │   ├── L1/                     #     {clip_key}_L1_1fps.mp4（fps 重采样）
│   │   │   ├── L2/                     #     {clip_key}_L2_w{ws}_{we}.mp4
│   │   │   └── L3/                     #     {clip_key}_L3_ev{id}_{cs}_{ce}.mp4
│   │   ├── frames/                     #   1fps 原始帧目录（annotate.py 使用）
│   │   ├── annotate.py                 #   VLM 三层级联标注（L1→L2→L3）
│   │   ├── annotate_check.py           #   L2/L3 质量审核
│   │   ├── extract_frames.py           #   1fps 抽帧
│   │   ├── prepare_clips.py            #   ffmpeg clip 准备（L1/L2/L3，含 L1 fps 重采样）
│   │   ├── prompts.py                  #   标注 + 训练 prompt 模板库
│   │   ├── datasets/                   #   生成的训练 JSONL
│   │   ├── build_dataset.py            #   [DEPRECATED] 已由 build_hier_data.py 替代
│   │   ├── sample_mixed_dataset.py     #   [DEPRECATED]
│   │   └── run_build.sh                #   [DEPRECATED]
│   │
│   ├── temporal_aot/                    # AOT 时序方向判断任务（从 seg 标注构建）
│   │   ├── build_aot_from_seg.py       #   ★ 主入口：从 seg 标注构建 4 种 AOT 任务
│   │   ├── build_event_aot_data.py     #   旧版：基于独立 VLM captioning（不推荐）
│   │   ├── annotate_event_captions.py  #   旧版：VLM caption 生成
│   │   ├── build_aot_mcq.py            #   旧版：MCQ 构建
│   │   ├── rebalance_aot_answers.py    #   答案重平衡工具
│   │   ├── prompts.py                  #   AOT prompt 模板
│   │   └── data/                       #   生成数据
│   │
│   └── event_logic/                     # 事件逻辑推理任务（从 L2 标注构建）
│       ├── build_l2_event_logic.py     #   ★ 主入口：V2T 任务（add/replace/sort）
│       ├── build_l2_event_logic_t2v.py #   T2V 变体
│       ├── annotate_l2_step_captions.py#   T2V 所需的 step caption 标注
│       ├── prompts.py                  #   event logic prompt 模板
│       ├── filter_bad_videos.py        #   视频健康校验
│       ├── merge_datasets.py           #   数据合并工具
│       └── data/                       #   生成数据
│
├── temporal_grounding/                  # 时序定位数据（独立来源，与 seg 无关）
│   ├── DESIGN.md
│   ├── build_dataset.py
│   └── convert_nocot_to_cot.py
│
├── youcookii_annotations_trainval.json  # YouCook2 原始标注
├── youcook2_train_easyr1.jsonl          # YouCook2 base 训练数据
├── youcook2_val_small.jsonl             # YouCook2 验证集
└── bad_videos.txt                       # 已知损坏视频列表
```

---

## 七、数据构建流程（标准流）

### 步骤 1：VLM 标注（一次性，已完成）

```bash
# 提取 1fps 帧
python youcook2_seg/youcook2_seg_annotation/extract_frames.py ...

# 三层级联标注（L1→L2→L3）
python youcook2_seg/youcook2_seg_annotation/annotate.py ...

# 质量审核
python youcook2_seg/youcook2_seg_annotation/annotate_check.py ...
```

### 步骤 2：生成物理 Clip 文件

```bash
# L1：fps 重采样（新流程，替代 warped 帧拼接）
python youcook2_seg/youcook2_seg_annotation/prepare_clips.py \
    --input /tmp/l1_raw.jsonl --output /tmp/l1_clipped.jsonl \
    --clip-dir /data/clips/L1 --l1-fps 1

# L2：128s 窗口截取
python youcook2_seg/youcook2_seg_annotation/prepare_clips.py \
    --input /tmp/l2_raw.jsonl --output /tmp/l2_clipped.jsonl \
    --clip-dir /data/clips/L2

# L3：event clip 截取（±5s padding）
python youcook2_seg/youcook2_seg_annotation/prepare_clips.py \
    --input /tmp/l3_raw.jsonl --output /tmp/l3_clipped.jsonl \
    --clip-dir /data/clips/L3
```

# L3：event clip 截取（±5s padding）
python youcook2_seg_annotation/prepare_clips.py \
    --input /tmp/l3_raw.jsonl --output /tmp/l3_clipped.jsonl \
    --clip-dir /data/clips/L3
```

### 步骤 3：构建各任务训练数据

```bash
# Hier-Seg（L1/L2/L3/L3_seg）
python local_scripts/hier_seg_ablations/build_hier_data.py \
    --annotation-dir /data/youcook2_seg/youcook2_seg_annotation/annotations \
    --clip-dir-l1 /data/clips/L1 --l1-fps 1 \
    --clip-dir-l2 /data/clips/L2 \
    --clip-dir-l3 /data/clips/L3 \
    --output-dir /data/output/hier_seg \
    --levels L1 L2 L3 L3_seg

# AOT（4 种任务）
python proxy_data/youcook2_seg/temporal_aot/build_aot_from_seg.py \
    --annotation-dir /data/youcook2_seg/youcook2_seg_annotation/annotations \
    --clip-dir-l2 /data/clips/L2 \
    --clip-dir-l3 /data/clips/L3 \
    --output-dir /data/output/aot

# Event Logic
python proxy_data/youcook2_seg/event_logic/build_l2_event_logic.py \
    --annotation-dir /data/youcook2_seg/youcook2_seg_annotation/annotations \
    --clips-dir /data/clips/L2 \
    --output /data/output/event_logic/train.jsonl
```

### 扩充数据

只需往 `youcook2_seg/youcook2_seg_annotation/annotations/` 加入新的 `*.json` 标注文件，重新运行步骤 2–3 即可。
共享接口 `shared/seg_source.py` 自动被三条流水线加载。

---

## 八、奖励函数设计

| problem_type | Reward 函数 | 策略 |
|---|---|---|
| `temporal_seg_hier_L1` / `L2` | F1-IoU（NMS + 匈牙利匹配） | 连续奖励，精度召回均衡 |
| `temporal_seg_hier_L3` | Position-Aligned Mean tIoU | 按位置对齐，无匹配需要 |
| `temporal_seg_hier_L3_seg` | F1-IoU | 同 L1/L2 |
| `seg_aot_*` | 精确匹配选择题 | 有 `<answer>` 且正确 → 1.0 |
| `add` / `replace` / `sort` | 选择题 / Jigsaw Displacement | — |

**文件位置**：
- `verl/reward_function/mixed_proxy_reward.py`（AOT / event logic）
- `verl/reward_function/youcook2_hier_seg_reward.py`（hier-seg L1/L2/L3）

---

## 九、视频健康校验（训练前必做）

训练中曾因损坏的 h264 视频导致 decord seek 卡死，引发 NCCL 超时。

```bash
# 模拟训练采样方式校验（--full-decode 必须开启）
python proxy_data/youcook2_seg/event_logic/filter_bad_videos.py \
    -i /path/to/train.jsonl \
    -o /path/to/train_clean.jsonl \
    --full-decode --workers 8 \
    --bad_list bad_videos.txt
```

| 参数 | 说明 |
|------|------|
| `--full-decode` | 模拟 fps=2.0 跳帧 seek（与训练完全对齐） |
| `--video-fps` | 目标采样帧率，与 `common.sh` 的 `VIDEO_FPS` 对齐 |
| `--max-frames` | 最大帧数，与 `common.sh` 的 `MAX_FRAMES` 对齐 |
| `--bad-list-input` | 复用已有坏视频列表，跳过解码 |
