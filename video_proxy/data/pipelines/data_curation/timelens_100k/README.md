# TimeLens-100K 筛选说明

这个目录记录 TimeLens-100K 作为候选视频池时的使用方式。实际代码在上一级 `curation/`，主入口是 `../run.sh`。

## 推荐命令

```bash
DATASET=timelens_100k \
INPUT=/path/to/timelens-100k.jsonl \
VIDEO_ROOT=/path/to/TimeLens-100K/video_shards \
MIN_DURATION=0 \
MAX_DURATION=60 \
TARGET_TOTAL=3000 \
BALANCED_TOTAL=1 \
bash video_proxy/data/pipelines/data_curation/run.sh
```

## 输出

默认输出到：

```text
video_proxy/data/pipelines/data_curation/results/timelens_100k/
├── duration_keep.jsonl
├── duration_summary.json
└── screen_keep.jsonl
```

如果启用 `LOCAL_SCORE=1`，还会生成 `screen_results.jsonl` 和 `screen_reject.jsonl`。
