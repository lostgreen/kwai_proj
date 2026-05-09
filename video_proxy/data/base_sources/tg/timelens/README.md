# TimeLens TG Rollout

这个目录负责把 TimeLens-100K 转成 Temporal Grounding rollout 数据，并按 IoU 难度筛选后补充到 TimeR1 train。

## 主要文件

- `build_rollout_dataset.py`：从 TimeLens 原始 JSONL 构建 TG rollout 输入。
- `run_rollout.sh`：调用离线 rollout 工具并计算 TG reward。
- `analyze_rollout.py`：统计 rollout 结果、IoU 和命中分布。
- `select_iou_range.py`：按 IoU 区间筛选可训练样本。
- `merge_with_time_r1.py`：把筛选后的 TimeLens 样本合并到 TimeR1 train。

## 常用命令

```bash
# 1. 构建 rollout 输入
python video_proxy/data/base_sources/tg/timelens/build_rollout_dataset.py \
  --input /path/to/timelens-100k.jsonl \
  --video-root /path/to/TimeLens-100K/videos \
  --output /path/to/timelens_rollout_input.jsonl

# 2. 运行 rollout
INPUT=/path/to/timelens_rollout_input.jsonl \
OUTPUT_ROOT=/path/to/timelens_rollout \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
NUM_GPUS=8 \
bash video_proxy/data/base_sources/tg/timelens/run_rollout.sh

# 3. 分析结果
python video_proxy/data/base_sources/tg/timelens/analyze_rollout.py \
  --input /path/to/timelens_rollout/rollout_report.jsonl \
  --output-dir /path/to/timelens_rollout/analysis

# 4. 按 IoU 区间筛选
python video_proxy/data/base_sources/tg/timelens/select_iou_range.py \
  --input /path/to/timelens_rollout/rollout_report.jsonl \
  --output /path/to/timelens_selected.jsonl \
  --min-iou 0.1 \
  --max-iou 0.5

# 5. 合并到 TimeR1 train
python video_proxy/data/base_sources/tg/timelens/merge_with_time_r1.py \
  --time-r1 /path/to/tg_timerft_max256s_validated.jsonl \
  --timelens /path/to/timelens_selected.jsonl \
  --output /path/to/tg_train_with_timelens.jsonl
```

## 输出

- rollout 输入：`timelens_rollout_input.jsonl`
- rollout 报告：`rollout_report.jsonl`
- IoU 筛选结果：`timelens_selected.jsonl`
- 合并后的 TG train：`tg_train_with_timelens.jsonl`
