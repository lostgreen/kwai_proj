# Segmentation Visualize

轻量本地可视化工具，用于查看 `youcook2_seg_annotation` 的三层时序标注结果，
也支持查看 `build_dataset.py` 产出的训练 JSONL（尤其是 L3）：

- `1fps` 帧条
- `Level 1 / Level 2 / Level 3` 三层时间轴
- segment 文本详情
- 基础诊断：gap / overlap / child outside parent

## Start

在仓库根目录执行：

```bash
bash data_visualization/segmentation_visualize/run.sh
```

也可以通过环境变量指定要预加载的目录：

```bash
ANNOTATION_DIR=/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations \
MAX_SAMPLES=200 \
PORT=8890 \
bash data_visualization/segmentation_visualize/run.sh
```

或直接加载 `build_dataset.py` 输出的训练 JSONL（通过 `DATA_PATH` 环境变量）：

```bash
# 可视化 L3 训练数据
DATA_PATH=/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/datasets/youcook2_hier_L3_train.jsonl \
PORT=8893 \
bash data_visualization/segmentation_visualize/run.sh

# 可视化 L2 训练数据（滑窗样本）
DATA_PATH=/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/datasets/youcook2_hier_L2_train.jsonl \
PORT=8892 \
bash data_visualization/segmentation_visualize/run.sh

# 可视化 L1 训练数据
DATA_PATH=/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/datasets/youcook2_hier_L1_train.jsonl \
PORT=8891 \
bash data_visualization/segmentation_visualize/run.sh
```

或手动启动：

```bash
python data_visualization/segmentation_visualize/server.py \
  --host 127.0.0.1 \
  --port 8890 \
  --static-dir data_visualization/segmentation_visualize \
  --annotation-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations \
  --max-samples 200 \
  --prefer-complete
```

打开：

`http://127.0.0.1:8890/`

如果通过 `run.sh`、`--annotation-dir` 或 `--data-path` 启动，服务会在启动时直接：

- 解析 annotation 数据
- 按优先级选择要预热的 clips
- 生成这些 clips 的 base64 帧条
- 把 `summary + all_details` 预注入页面 HTML
- 在终端打印预加载进度条

这样页面打开后就能直接渲染，使用方式更接近 `rollout_visualization`。

## Usage

页面顶部填写数据路径，例如：

```text
proxy_data/youcook2_seg_annotation/annotations
```

也支持绝对路径，例如：

```text
/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations
```

也支持直接加载 `build_dataset.py` 输出的 JSONL，例如：

```text
proxy_data/youcook2_seg_annotation/youcook2_hier_L3_train.jsonl
```

加载后：

- 左侧是 clip 列表
- 右侧是当前 clip 的帧条和三层时间轴
- hover 任意 segment 会联动高亮
- click segment 或 frame 会在下方锁定详情
- 帧条图片由后端在 `GET /api/clip/<clip_key>` 中直接内嵌成 base64 data URL，不再逐帧二次请求

如果已经用启动参数预加载，页面会自动渲染，不需要再手动输入路径。

## Preload Strategy

- `MAX_SAMPLES=0` 表示预加载全部 clips
- `MAX_SAMPLES>0` 时，只预热前 `N` 个 clip 的详情和 base64 帧条
- 开启 `--prefer-complete` 或 `PREFER_COMPLETE=1` 时，预热顺序会优先选择同时具备 `level1 + level2 + level3` 的样本

注意：

- 这里的 `max_samples` 只影响“启动时预注入到 HTML 的样本数”
- clip 列表和元数据仍然可以保留全量
- 没有被预热的 clip，后续点开时仍可由后端按需生成

## API

- `GET /api/load-data?data_path=...`
- `GET /api/state`
- `GET /api/clips?search=...`
- `GET /api/clip/<clip_key>`
