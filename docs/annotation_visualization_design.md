# Annotation Visualization Design

## Goal

为 `proxy_data/youcook2_seg_annotation` 这套三层时序标注数据设计一个轻量、可本地启动的可视化系统，核心体验是：

- 顶部展示视频基础信息与全局导航
- 中部展示 `1fps` 帧条
- 帧条下方同步展示 `Level 1 / Level 2 / Level 3` 三层时间轴
- 右侧或下方展示当前选中 segment 的文本标注内容
- 支持 hover / click 联动，快速检查时间边界与层级归属是否合理

该系统优先适配如下单个样本 JSON 结构：

- `clip_key`
- `clip_duration_sec`
- `n_frames`
- `frame_dir`
- `level1.macro_phases`
- `level2.meso_steps`
- `level3.key_state_chunks`

样例可参考：[/Users/lostgreen/Downloads/20260308_115202_results.json](/Users/lostgreen/Downloads/20260308_115202_results.json)

## Product Shape

建议做成一个“零构建”的本地 Web 工具：

- 后端：Python `http.server` 风格的轻量 API
- 前端：单页 `HTML + CSS + Vanilla JS`
- 复用现有模式：[`rollout_visualization/server.py`](/Users/lostgreen/Desktop/Codes/kwai_proj/rollout_visualization/server.py) + [`rollout_visualization/index.html`](/Users/lostgreen/Desktop/Codes/kwai_proj/rollout_visualization/index.html)

这样不需要额外引入 React / Node 依赖，部署和迁移都简单，特别适合服务器上直接跑。

## Core Screen

建议主页面采用三栏但偏中轴的布局：

```text
+--------------------------------------------------------------------------------------+
| Header: clip_key | duration | n_frames | source_mode | prev/next | 搜索 | 视图开关     |
+--------------------------------------------------------------------------------------+
| Overview Strip                                                                       |
| [L1 summary cards] [L2 count] [L3 count] [coverage] [gaps]                           |
+--------------------------------------------------------------------------------------+
| 1fps Frames                                                                          |
| [0001][0002][0003][0004]...[0331]  <- 横向可滚动，支持缩放                            |
+--------------------------------------------------------------------------------------+
| Level 1 Timeline                                                                     |
| [Topping Preparation----------][Pizza Assembly-------------------][Bake][Serve------]|
+--------------------------------------------------------------------------------------+
| Level 2 Timeline                                                                     |
| [Mince garlic][Mix oil][Toss tomato]...[Serve slice]                                 |
+--------------------------------------------------------------------------------------+
| Level 3 Timeline                                                                     |
| [Peel][Chop][Transfer][Mix]...[Plate slice]                                          |
+--------------------------------------------------------------------------------------+
| Bottom Area                                                                          |
| Left: 当前选中 segment 文本详情   | Right: 父子关系 / keywords / pre-post state / raw |
+--------------------------------------------------------------------------------------+
```

## Main Interactions

### 1. 时间同步

- 帧条和三层时间轴共享同一个横向滚动容器或滚动位置
- 用户拖动帧条时，`L1/L2/L3` 一起滚动
- 用户点击某个时间段时，帧条自动滚动到对应位置并高亮对应帧

### 2. Hover 联动

- hover `L1 phase`：
  - 高亮该 phase 覆盖的帧范围
  - 同时高亮其下属的所有 `L2 step`
  - 右侧显示 `phase_name + narrative_summary`
- hover `L2 step`：
  - 高亮对应帧
  - 弱化其它 step
  - 右侧显示 `instruction + visual_keywords`
- hover `L3 chunk`：
  - 高亮对应帧
  - 显示 `sub_action / pre_state / post_state`

### 3. Click 锁定

- 单击某个 segment 后进入锁定态
- 底部详情区固定展示该 segment
- 再点空白区域或关闭按钮取消锁定

### 4. 帧级检查

- 点击某一帧时：
  - 弹出大图预览
  - 显示该帧落在哪些 `L1/L2/L3` segment 中
  - 显示 `frame_index -> timestamp`

## Recommended Visual Language

为三层语义建立稳定色系，避免页面像“调试台”一样过乱：

- `L1`: 深蓝 / 青绿色块，强调宏观阶段
- `L2`: 橙色 / 金色块，强调可执行步骤
- `L3`: 玫红 / 紫红色块，强调细粒度状态变化

建议样式方向：

- 背景：浅暖灰或浅蓝灰，不要纯白
- 时间轴块：圆角胶囊条
- 文本卡片：半透明白卡 + 细边框
- 字体：
  - 中文：`PingFang SC` / `Microsoft YaHei`
  - 英文和时间：`JetBrains Mono`

## Data Mapping

### Time Conversion

页面内部统一转成秒处理：

- `00:09 -> 9`
- `03:58 -> 238`

若 `frames` 为 `1fps` 且文件名为 `0001.jpg`：

- `frame 1 ≈ 00:01`
- `frame 9 ≈ 00:09`

建议采用现有标注脚本同样的映射约定，保持一致性：

- 时间范围 `[start_sec, end_sec]`
- 帧命中规则：`start_sec <= frame_idx <= end_sec`

### Derived Structures

前端加载后，把数据整理成统一 timeline item：

```json
{
  "id": "l2-7",
  "level": 2,
  "label": "Sprinkle shredded mozzarella cheese over the pizza",
  "startSec": 142,
  "endSec": 161,
  "parentId": "l1-2",
  "meta": {
    "visual_keywords": ["shredded mozzarella", "sprinkling"]
  }
}
```

### Parent-child Index

需要在前端预构建：

- `phase_id -> meso_steps[]`
- `step_id -> key_state_chunks[]`
- `frame_idx -> [matched segments]`

这样 hover 和 click 联动才能顺畅。

## Information Architecture

### Header

- `clip_key`
- `duration`
- `n_frames`
- `annotation range`
- `source_mode`
- 上一个 / 下一个样本
- 搜索框：按 `clip_key` 跳转

### Left Filter Drawer

如果后续要做成多样本浏览器，左侧建议支持：

- clip 列表
- 仅看有 `level3` 的样本
- 仅看存在 gap / overlap 的样本
- 按视频时长排序
- 按 phase 数量排序

### Bottom Detail Tabs

建议详情区使用 tab：

- `Summary`
- `Level 1`
- `Level 2`
- `Level 3`
- `Raw JSON`

其中：

- `Summary` 展示当前选中时间段的父子层级摘要
- `Raw JSON` 方便直接检查原始标注

## Quality-of-life Features

### 1. Gap / Overlap Diagnostics

系统自动检测每层内部：

- segment overlap
- large gap
- unordered timestamps
- child 超出 parent 范围

并在 UI 上用小告警标记：

- 黄色：gap 较大
- 红色：overlap / 越界

### 2. Sampling Overlay

把 `level1._sampling.sampled_frame_indices` 也作为一层稀疏标记显示出来：

- 在 1fps 帧条上打小点
- 让用户能看到 `L1` 实际用的是哪些 `0.5fps` 帧

这会非常适合检查“`L1` 粗标是否因为采样遗漏了关键边界”。

### 3. Keyboard Shortcuts

- `A / D`：上一帧 / 下一帧
- `J / L`：往前 / 往后跳 10 秒
- `1 / 2 / 3`：只显示 L1 / L2 / L3
- `F`：聚焦当前选中 segment
- `R`：打开 raw JSON

### 4. Screenshot Export

支持导出当前视窗：

- `PNG`
- 或导出一个 JSON snapshot，保存当前选中状态

## Minimal API Design

建议新增独立目录，例如：

- `annotation_visualization/server.py`
- `annotation_visualization/index.html`

后端 API 可以保持非常简单：

### `GET /api/clips`

返回样本列表摘要：

```json
[
  {
    "clip_key": "abc",
    "duration_sec": 324,
    "n_frames": 331,
    "has_level1": true,
    "has_level2": true,
    "has_level3": true
  }
]
```

### `GET /api/clip/<clip_key>`

返回单样本完整 JSON。

### `GET /api/frame/<clip_key>/<frame_idx>`

返回缩略图或原图。

建议服务端做两档：

- timeline 缩略图：宽度 `120px`
- modal 大图：原始 JPG

### `GET /api/diagnostics/<clip_key>`

返回派生诊断：

- gap
- overlap
- child-outside-parent
- coverage ratio

## Rendering Strategy

### Frame Strip

每帧固定宽度，例如：

- 宽 `88px`
- 高 `54px`
- 间距 `4px`

长视频下不要一次性渲染全部大图，建议：

- 默认渲染可视范围附近的帧
- 其它帧只渲染占位骨架

如果先求稳，也可以先做非虚拟列表，因为单视频几百帧仍能跑。

### Timeline Lanes

每层使用独立 lane：

- 纵向高度固定
- 横向按时间比例定位
- segment 以绝对定位绘制

每个 segment block 至少显示：

- 名称
- `start-end`

过窄时只显示短标签，完整文本放 tooltip。

## Suggested Milestones

### Phase 1: 单样本查看器

- 输入一个 annotation JSON 路径
- 加载 `frame_dir`
- 显示帧条 + 三层时间轴 + 详情卡

### Phase 2: 多样本列表

- 浏览整个 `annotations/` 目录
- 支持按 `clip_key` 搜索
- 切换上下样本

### Phase 3: 诊断与导出

- 自动质量检查
- 截图导出
- 批量筛选问题样本

## Best First Implementation Choice

为了最快落地，推荐直接基于现有原型扩展，而不是新起一个前端工程：

1. 复制一份 [`rollout_visualization/server.py`](/Users/lostgreen/Desktop/Codes/kwai_proj/rollout_visualization/server.py)
2. 改成读取 `annotations/*.json`
3. 在新的 `index.html` 中实现：
   - 帧条
   - 三层 lane
   - 详情面板
4. 第二阶段再补多样本切换和诊断

## Why This Fits Current Data Well

这套设计和你当前数据结构是天然匹配的：

- `frame_dir` 已经提供 1fps 原始检查素材
- `level1/2/3` 都是明确的时间段结构
- `level2._segment_calls` 与 `level3._segment_calls` 还能反查每次分层调用用了哪些帧
- `level1._sampling` 还能额外显示粗标采样点

换句话说，这个系统不只是“展示结果”，还可以直接检查：

- 上一层时间范围有没有把下一层裁准
- 某个 step / chunk 的边界是不是对着正确帧
- 分层采样是否稳定

## Next Step

如果继续做实现，建议下一步直接产出一个可运行的 MVP：

- 页面能打开单个 `annotation.json`
- 自动读取 `frame_dir`
- 渲染 `1fps` 帧条
- 渲染 `L1/L2/L3` 三层时间轴
- 点击任意 segment 查看文本详情

这是最小但已经非常有用的一版。
