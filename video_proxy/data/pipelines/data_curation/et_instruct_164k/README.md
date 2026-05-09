# ET-Instruct-164K 筛选说明

这个目录记录 ET-Instruct-164K 作为候选视频池时的使用方式。实际代码在上一级 `curation/`，主入口是 `../run.sh`。

## 推荐命令

```bash
DATASET=et_instruct_164k \
INPUT=/path/to/et_instruct_164k_txt.json \
VIDEO_ROOT=/path/to/ET-Instruct-164K/videos \
MIN_DURATION=60 \
MAX_DURATION=240 \
bash video_proxy/data/pipelines/data_curation/run.sh
```

## 输出

默认输出到：

```text
video_proxy/data/pipelines/data_curation/results/et_instruct_164k/
├── duration_keep.jsonl
├── duration_summary.json
└── screen_keep.jsonl
```

`screen_keep.jsonl` 是后续 `proxy_construction/annotation/` 的常用输入。
