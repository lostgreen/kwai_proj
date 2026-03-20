# Unified Data Visualization — Design Document

## 1. 目标

将原本碎片化的 `segmentation_visualize/` 升级为一个**统一可视化系统**，直接放在 `data_visualization/` 下，支持同一份 server + 同一个 UI 处理三类数据：

| 视图 | 数据来源 | 核心功能 |
|------|---------|---------|
| **Seg** | 现有 annotation JSON / seg JSONL | 分层时序段条 + 帧条（现有功能原样保留） |
| **AoT Caption** | `caption_pairs.jsonl` + `aot_event_manifest.jsonl` | 每个 event clip 的 forward/reverse/shuffle 视频帧条 + VLM caption 并排对比 |
| **AoT MCQ** | `v2t_train.jsonl` / `t2v_train.jsonl` / `4way_*.jsonl` | 每题的视频帧条 + A/B/C/D 选项卡片 + 正确答案高亮 |

---

## 2. 文件结构

```
data_visualization/
├── DESIGN.md          ← 本文档
├── server.py          ← 统一后端（继承 segmentation_visualize/server.py 全部逻辑 + 新增两类 store）
├── index.html         ← 统一前端（三 Tab 页，复用帧条 CSS/JS 组件）
└── run.sh             ← 快捷启动脚本
```

旧的 `segmentation_visualize/` 目录保持原样，不删除（向后兼容）。

---

## 3. 数据模型

### 3.1 Seg 视图（继承现有）

输入：annotation JSON 目录 / seg JSONL 文件  
后端 Store：`SegmentationStore`（原有，不改）  
显示：三层时序 Lane + 帧条 + 诊断面板

### 3.2 AoT Caption 视图

输入：`caption_pairs.jsonl`（必需）+ 可选 `aot_event_manifest.jsonl`

每条 `caption_pairs` 记录字段：

```jsonc
{
  "clip_key": "GLd3aX16zBg_event00",
  "forward_caption": "...",
  "forward_confidence": 0.92,
  "forward_direction_clear": true,
  "reverse_caption": "...",
  "reverse_confidence": 0.88,
  "reverse_direction_clear": true,
  "shuffle_caption": "...",     // optional
  "shuffle_confidence": 0.76,   // optional
  "is_different": true
}
```

manifest 补充字段（可选）：

```jsonc
{
  "clip_key": "...",
  "forward_video_path": "/path/to/fwd.mp4",
  "reverse_video_path": "/path/to/rev.mp4",
  "shuffle_video_path": "/path/to/shuf.mp4",
  "start_sec": 90, "end_sec": 102,
  "sentence": "spread margarine on two slices ...",
  "recipe_type": "113",
  "shuffle_segment_sec": 2.0
}
```

后端：`AoTCaptionStore`   
- `load(caption_pairs_path, manifest_path?)` → dict of clip records   
- `get_clip(clip_key)` → `AoTCaptionClip`，包含采样帧 data-URLs for each direction

UI 布局（每条记录）：
```
┌──────────────────── clip header (clip_key, sentence, duration, recipe) ───────────┐
│  [direction_clear badge]  [confidence badge]  [is_different badge]                │
├──────────────┬────────────────┬─────────────────────────────────────────────────── │
│  FORWARD     │  REVERSE        │  SHUFFLE (if present)                             │
│  帧条(2fps)  │  帧条(1fps)     │  帧条(2fps, segment-aware)                        │
│  caption文字 │  caption文字    │  caption文字                                       │
│  conf badge  │  conf badge     │  conf badge                                        │
└──────────────┴────────────────┴────────────────────────────────────────────────── ┘
```

### 3.3 AoT MCQ 视图

输入：任意 MCQ JSONL 文件（`aot_v2t` / `aot_t2v` / `aot_4way_v2t` / `aot_4way_t2v`）

每条记录字段（通用）：

```jsonc
{
  "problem_type": "aot_v2t",
  "prompt": "Watch the video carefully...\nA. ...\nB. ...",
  "answer": "A",
  "videos": ["/path/to/clip.mp4"],
  "metadata": {
    "clip_key": "...",
    "forward_caption": "...",
    "reverse_caption": "...",
    "video_direction": "forward",
    ...
  }
}
```

后端：`AoTMCQStore`  
- 从 JSONL line-by-line 读取，每条记录为一个 `MCQRecord`  
- 为每个视频路径抽帧（server 端用 decord/OpenCV 抽 ≤16 帧），生成 data-URL

UI 布局（每题）：
```
┌─────────────────── MCQ 题头 ─────────────────────────────────────────────────────┐
│  problem_type badge   clip_key   video_direction                                   │
├────────────────────────────────── 视频帧条 ──────────────────────────────────────┤
│  [frame] [frame] [frame] ...（复用 seg 的帧条样式，同一 CSS class）               │
├───────────────────────────────── 选项卡片区 ─────────────────────────────────────┤
│  ┌──── A ────┐  ┌──── B ────┐  ┌──── C (4way) ─┐  ┌──── D (4way) ─┐           │
│  │ caption  │  │ caption  │  │  caption       │  │  caption       │           │
│  │ ✓ CORRECT│  │          │  │                │  │                │           │
│  └──────────┘  └──────────┘  └─────────────── ┘  └─────────────── ┘           │
│  (T2V: 选项为视频帧条，而非文字)                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

V2T（选项为文字 caption）vs T2V（选项为视频帧条）在同一 UI 自动切换渲染。

---

## 4. 后端 API 设计

在现有 `/api/` 路由基础上增加：

```
# AoT Caption
GET /api/aot-caption/load-data?caption_pairs=...&manifest=...
GET /api/aot-caption/state
GET /api/aot-caption/clips?search=
GET /api/aot-caption/clip/:clip_key

# AoT MCQ
GET /api/aot-mcq/load-data?data_path=...
GET /api/aot-mcq/state
GET /api/aot-mcq/records?search=
GET /api/aot-mcq/record/:record_id
```

帧提取：  
- 所有视频帧提取统一在 server 端按路径缓存（`video_path → list[data-URL]`）  
- 使用 `decord.VideoReader` 主路径，失败 fallback 到 `subprocess ffmpeg -vf fps=1` 管道读取  
- 最多提取 24 帧，JPEG 压缩后 base64 内嵌到 JSON

---

## 5. 前端设计

### 5.1 Tab 导航

顶部三 Tab，切换视图，各自维护独立状态：

```
[   Segmentation   ]  [  AoT Caption  ]  [  AoT MCQ  ]
```

Tab 间切换不重置已加载数据，URL hash 记录当前 Tab（`#seg` / `#caption` / `#mcq`）。

### 5.2 共用组件（复用 seg 现有实现）

| 组件 | CSS class | 用途 |
|------|-----------|------|
| 帧条 | `.frame-row` / `.frame-card` / `.frame-thumb` | 三个视图均复用 |
| 帧条滚动列 | `.sync-shell` / `.sync-inner` | 复用 |
| Badge | `.badge` `.ok` `.warning` | 复用 |
| Card | `.card` | 复用 |
| Sidebar clip list | `.clip-list` / `.clip-row` | AoT Caption 视图复用 |

### 5.3 AoT Caption 专属组件

```css
.caption-trio          /* 三列并排容器 (grid 3fr 3fr 3fr) */
.caption-col           /* 单列：帧条 + caption 文字 */
.caption-text          /* caption 文字气泡样式 */
.direction-badge       /* FORWARD / REVERSE / SHUFFLE 颜色标记 */
.confidence-bar        /* 0-1 横向置信度条 */
```

### 5.4 AoT MCQ 专属组件

```css
.mcq-option-grid       /* 2列 or 4列 option 卡片容器 */
.mcq-option-card       /* 单个选项卡片 */
.mcq-option-card.correct   /* 正确答案高亮（绿色边框 + 背景） */
.mcq-option-card.wrong     /* 错误选项（灰色/50%透明度） */
.option-letter         /* A / B / C / D 大字母标识 */
.mcq-type-badge        /* V2T / T2V / 4way badge */
```

---

## 6. 视频帧提取策略

服务端抽帧，不依赖 `frame_dir`（AoT 数据没有预提取帧目录）：

```python
def extract_video_frames(video_path, max_frames=16, target_fps=None) -> list[str]:
    # 1. 尝试 decord.VideoReader
    # 2. fallback: ffmpeg -vf fps=1 -f image2pipe | base64
    # 每帧 resize 到 max_width=200px，JPEG 编码
    # 返回 data-URL list
```

缓存策略：`video_path → (frames_list, mtime)` 进程内 dict，按 mtime 失效。

T2V 4-way 记录有 4 个视频，每个各提取帧，response 体积会较大（约 4×16×10KB = 640KB），在可接受范围。

---

## 7. 实施步骤

### Step 1：新建 `data_visualization/server.py`
- 完整移植 `segmentation_visualize/server.py` 全部逻辑
- 新增 `VideoFrameExtractor` 类（decord + ffmpeg fallback）
- 新增 `AoTCaptionStore` 类
- 新增 `AoTMCQStore` 类
- 新增 `/api/aot-caption/*` 和 `/api/aot-mcq/*` 路由
- CLI 保留原有参数，新增 `--caption-pairs`、`--mcq-data`

### Step 2：新建 `data_visualization/index.html`
- 顶部 Tab 导航（Seg / AoT Caption / AoT MCQ）
- 完整移植现有 Seg 视图（所有 CSS + JS）
- 新增 AoT Caption 视图（`.caption-trio`、`.caption-text` 等）
- 新增 AoT MCQ 视图（`.mcq-option-grid`、`.mcq-option-card` 等）
- 共用帧条组件，JS 按当前 Tab 路由渲染

### Step 3：`data_visualization/run.sh`

### Step 4：语法检查 + 本地测试

### Step 5：git commit + push

---

## 8. 关键设计决策

1. **不合并 store**：SegmentationStore / AoTCaptionStore / AoTMCQStore 三个独立类，共用 `ThreadingHTTPServer` 实例（server 上挂三个 store 属性）
2. **帧条坐标系**：AoT 视图的帧条以「帧序号 0-N」展示，不用时码，与 seg 的「1fps = second」区分
3. **T2V vs V2T 自适应**：前端根据 `problem_type` 字段自动切换 option 渲染（文字 vs 帧条），不需要用户手动配置
4. **数据量上限**：MCQ 视图侧边栏最多显示 500 条，超出时提示；clip 帧缓存按 LRU 限制 200 个 video path
5. **旧路径兼容**：`/api/load-data`、`/api/state`、`/api/clip/*` 保持不变，seg 视图不感知新路由
