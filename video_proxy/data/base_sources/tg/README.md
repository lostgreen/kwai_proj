# Temporal Grounding 基础数据

这里管理 Temporal Grounding 的基础数据构建。当前拆成两个来源：

- `time_r1/`：TimeR1/TimeRFT 与 TVGBench 的裁剪、构建、验证。
- `timelens/`：TimeLens-100K 的 rollout、IoU 筛选和与 TimeR1 train 合并。

公共格式检查和转换工具放在本目录根部。

## 主要文件

- `time_r1/run_pipeline.sh`：TimeR1/TimeRFT 与 TVGBench 的主入口。
- `time_r1/build_dataset.py`：把原始标注转成 TG JSONL。
- `time_r1/trim_videos.py`：按标注裁剪超长视频。
- `time_r1/validate_videos.py`：检查视频路径和时间戳有效性。
- `timelens/run_rollout.sh`：TimeLens rollout 入口。
- `common.py`：TG 共享字段、读写和 prompt 工具。
- `check_format.py`：检查 TG prompt 和字段格式。
- `rewrite_format.py`：重写已有 TG JSONL 的 prompt 格式。
- `convert_nocot_to_cot.py`：把 no-CoT 样本转成 CoT 样式。

## TimeR1/TVGBench 一键运行

```bash
TIMERFT_JSON=/path/to/train_2k5.json \
TVGBENCH_JSON=/path/to/tvgbench.json \
VIDEO_ROOT=/path/to/TimeR1-Dataset \
OUTPUT_DIR=video_proxy/data/base_sources/tg/data \
MAX_DURATION=256 \
TRIM_WORKERS=8 \
bash video_proxy/data/base_sources/tg/time_r1/run_pipeline.sh
```

如果视频已经裁剪好，可以跳过裁剪或验证：

```bash
bash video_proxy/data/base_sources/tg/time_r1/run_pipeline.sh --skip-trim
bash video_proxy/data/base_sources/tg/time_r1/run_pipeline.sh --skip-validate
```

## 格式检查与转换

```bash
# 检查 prompt 和时间戳格式
python video_proxy/data/base_sources/tg/check_format.py \
  video_proxy/data/base_sources/tg/data/tg_timerft_max256s_validated.jsonl

# 重写 prompt 格式
python video_proxy/data/base_sources/tg/rewrite_format.py \
  --input /path/to/input.jsonl \
  --output /path/to/output.jsonl

# no-CoT 转 CoT
python video_proxy/data/base_sources/tg/convert_nocot_to_cot.py \
  --input /path/to/nocot.jsonl \
  --output /path/to/cot.jsonl
```

## 输出

- `tg_timerft_max256s_validated.jsonl`：训练侧 TG base 来源。
- `tg_tvgbench_max256s_validated.jsonl`：验证侧 TG 采样来源。
- `timelens/` 产生的筛选结果可以通过 `merge_with_time_r1.py` 合并进 TimeR1 train。

## 注意

旧路径 `video_proxy/data/pipelines/temporal_grounding/` 已迁移到本目录。新增 TG 基础数据逻辑请优先放在 `time_r1/`、`timelens/` 或 `common.py`。
