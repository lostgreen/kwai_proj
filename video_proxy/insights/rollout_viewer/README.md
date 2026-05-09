# Rollout Viewer

轻量 rollout 查看器。这个工具只做三件事：

- 按 `uid` 聚合同一条样本的多次模型 rollout。
- 按 1fps 展示视频帧，便于快速扫视频内容。
- 展示选择题选项、ground truth 和完整模型 response。

它不再承担 ablation comparison 或按任务类型拆分的复杂分析视图。

## 启动

```bash
bash video_proxy/insights/rollout_viewer/run.sh
```

打开：

```text
http://localhost:8890/
```

也可以启动时预加载数据：

```bash
PORT=9000 \
ROLLOUT_DIR=checkpoints/<exp>/rollouts \
LOG_FILE=checkpoints/<exp>/experiment_log.jsonl \
bash video_proxy/insights/rollout_viewer/run.sh
```

## 手动启动

```bash
python video_proxy/insights/rollout_viewer/server.py \
  --host 0.0.0.0 \
  --port 8765 \
  --static-dir video_proxy/insights/rollout_viewer
```

## 数据字段

后端兼容常见 rollout JSONL 字段：

- `uid`
- `step` / `phase`
- `problem_type`
- `prompt`
- `ground_truth`
- `response`
- `reward`
- `metadata.options_list` / `metadata.options` / prompt 内的 `A. ...` 选项
- `multi_modal_source.frames_base64`
- `multi_modal_source.videos`
- `video_paths`
- `image_paths`

视频文件会在服务端抽 1fps 缩略帧。没有 `decord` 时，后端会 fallback 到 `qwen_vl_utils` 的 1fps 视频读取路径。
