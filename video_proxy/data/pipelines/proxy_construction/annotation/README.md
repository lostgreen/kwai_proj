# Annotation：层级标注

本目录负责把候选视频转换成层级事件标注 JSON。主流程是先抽取标注帧，再检测 scene boundary，最后用 VLM 做 scene-first 层级标注。

## 文件说明

- `run_annotation_pipeline.sh`：推荐的一键入口。
- `extract_annotation_frames.py`：从候选视频抽取标注帧。
- `detect_scene_boundaries.py`：检测 scene boundary，作为标注硬锚点。
- `build_hierarchical_annotations.py`：调用 VLM 生成 L1/L2/L3 层级标注。
- `annotation_schema.py`：标注 schema、领域 taxonomy、archetype/topology 映射和 prompt。

## 运行命令

小规模冒烟测试：

```bash
JSONL=video_proxy/data/pipelines/data_curation/results/et_instruct_164k/screen_keep.jsonl \
DATA_ROOT=/path/to/proxy_annotation_root \
LIMIT=5 \
bash video_proxy/data/pipelines/proxy_construction/annotation/run_annotation_pipeline.sh
```

正式运行示例：

```bash
JSONL=video_proxy/data/pipelines/data_curation/results/et_instruct_164k/screen_keep.jsonl \
DATA_ROOT=/path/to/proxy_annotation_root \
MODEL=pa/gmn-2.5-pr \
WORKERS=8 \
LIMIT=0 \
EXTRACT_FPS=1 \
bash video_proxy/data/pipelines/proxy_construction/annotation/run_annotation_pipeline.sh
```

单步运行也可以，但一般不建议跳过一键脚本：

```bash
python video_proxy/data/pipelines/proxy_construction/annotation/extract_annotation_frames.py \
  --jsonl /path/to/screen_keep.jsonl \
  --output-dir /path/to/proxy_annotation_root/frames \
  --fps 1 \
  --workers 8 \
  --limit 5
```

## 输出

```text
$DATA_ROOT/
├── frames/
├── annotations/
└── logs/
```

每个候选视频对应一个 annotation JSON。脚本尽量幂等，已有中间结果会复用；单个样本失败会写日志并继续处理后续样本。

## 注意

如果使用 `SCENE_DETECTOR=transnet`，需要本机 Python 或 `TRANSNET_VENV` 中安装 TransNetV2 依赖。无法使用时可以切换到 PySceneDetect 路径。
