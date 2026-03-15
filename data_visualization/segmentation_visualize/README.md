# Segmentation Visualize

轻量本地可视化工具，用于查看 `youcook2_seg_annotation` 的三层时序标注结果：

- `1fps` 帧条
- `Level 1 / Level 2 / Level 3` 三层时间轴
- segment 文本详情
- 基础诊断：gap / overlap / child outside parent

## Start

在仓库根目录执行：

```bash
bash data_visualization/segmentation_visualize/run.sh
```

或手动启动：

```bash
python data_visualization/segmentation_visualize/server.py \
  --host 127.0.0.1 \
  --port 8787 \
  --static-dir data_visualization/segmentation_visualize
```

打开：

`http://127.0.0.1:8787/`

## Usage

页面顶部填写 `annotation_dir`，例如：

```text
proxy_data/youcook2_seg_annotation/annotations
```

也支持绝对路径，例如：

```text
/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations
```

加载后：

- 左侧是 clip 列表
- 右侧是当前 clip 的帧条和三层时间轴
- hover 任意 segment 会联动高亮
- click segment 或 frame 会在下方锁定详情

## API

- `GET /api/load-data?annotation_dir=...`
- `GET /api/state`
- `GET /api/clips?search=...`
- `GET /api/clip/<clip_key>`
- `GET /api/frame/<clip_key>/<frame_idx>?mode=thumb|full`
