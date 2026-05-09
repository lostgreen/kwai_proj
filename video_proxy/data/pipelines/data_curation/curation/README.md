# 筛选实现模块

这里放 `data_curation/run.sh` 调用的 Python 实现。

## 文件职责

- `sources.py`：读取不同原始数据集并转成统一候选记录。
- `duration_filter.py`：按时长、来源上限、总量上限和均衡采样筛选。
- `local_score.py`：调用本地 VLM 对候选视频做二次打分。
- `io.py`：JSON/JSONL 读写与轻量工具函数。

## 直接运行

```bash
python -m video_proxy.data.pipelines.data_curation.curation.duration_filter \
  --dataset et_instruct_164k \
  --input /path/to/et_instruct_164k_txt.json \
  --video-root /path/to/videos \
  --output-dir /path/to/results \
  --min-duration 60 \
  --max-duration 240

python -m video_proxy.data.pipelines.data_curation.curation.local_score \
  --input-jsonl /path/to/results/duration_keep.jsonl \
  --output-jsonl /path/to/results/screen_results.jsonl \
  --keep-jsonl /path/to/results/screen_keep.jsonl \
  --reject-jsonl /path/to/results/screen_reject.jsonl \
  --model-path /path/to/Qwen3-VL-4B-Instruct
```

## 产物

`duration_filter.py` 输出 `duration_keep.jsonl` 和 `duration_summary.json`。`local_score.py` 输出 `screen_results.jsonl`、`screen_keep.jsonl` 和 `screen_reject.jsonl`。
