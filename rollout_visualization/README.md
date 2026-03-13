# Rollout Visualization Bundle

这个文件夹是完整可视化系统打包版，包含：

- `server.py`：后端服务（读取 rollout、按 uid 聚合、生成帧条数据）
- `index.html`：前端页面（帧条 + GT/Pred 时间轴 + n 次 rollout 对比）

## 1) 一键启动（推荐）

```bash
bash rollout_visualization/run.sh
```

可通过环境变量覆盖默认值：

```bash
HOST=0.0.0.0 PORT=9000 \
ROLLOUT_DIR=checkpoints/<exp>/rollouts \
LOG_FILE=checkpoints/<exp>/experiment_log.jsonl \
bash rollout_visualization/run.sh
```

## 2) 手动启动方式

在仓库根目录运行：

```bash
python rollout_visualization/server.py --host 0.0.0.0 --port 8765 --static-dir rollout_visualization
```

浏览器访问：

`http://<your-host>:8765/`

## 3) 页面里填写的数据路径

- `rollout_dir`：例如 `checkpoints/<exp_name>/rollouts`
- `log_file`（可选）：例如 `checkpoints/<exp_name>/experiment_log.jsonl`

点击“加载后端数据”后即可查看。

## 4) 你会看到什么

- 顶部：全局统计 + reward 趋势 + 任务分布
- 左侧：uid group 列表（step/task/search 过滤）
- 右侧：
  - 输入帧条（优先显示 base64 帧）
  - prompt / ground truth
  - 同一 uid 的 `n` 次 rollout 对比
  - `temporal_seg`：每次 rollout 的 GT/Pred 时间轴
  - 其他任务：输入输出与 proxy 简析

## 5) 数据兼容说明

后端优先使用 rollout 中的以下字段（如果存在）：

- `multi_modal_source`
- `multi_modal_source.frames_base64`（线上推荐字段，优先级最高）
- `video_paths`
- `image_paths`
- `temporal_segments`

如果 `temporal_segments` 缺失，会自动从文本中的 `[start, end]` 片段解析。

默认主字段约定：

- 分组键：`uid`
- 标注字段：`ground_truth`

## 6) 推荐的训练配置

为更好的可视化效果，建议打开：

```yaml
trainer:
  save_rollout_to_file: true
  save_rollout_include_multimodal: true
  save_rollout_include_timeline: true
```
