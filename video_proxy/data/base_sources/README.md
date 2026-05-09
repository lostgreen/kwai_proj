# 基础数据源

`base_sources/` 负责把外部已有数据集整理成 VideoProxy 可直接混合的基础 JSONL。这里的代码不构造新的代理任务，只做清洗、裁剪、格式统一、rollout 过滤和可视化筛选。

## 目录

- `tg/`：Temporal Grounding 基础数据，包含 TimeR1/TimeRFT、TVGBench 与 TimeLens rollout 补充。
- `mcq/`：LLaVA-Video-178K MCQ 基础数据，包含 MCQ 解析、rollout、筛选和可视化。

## 常用命令

```bash
# TG: 处理 TimeR1/TimeRFT 与 TVGBench
bash video_proxy/data/base_sources/tg/time_r1/run_pipeline.sh

# TG: 使用 TimeLens 做 rollout 扩充
bash video_proxy/data/base_sources/tg/timelens/run_rollout.sh

# MCQ: 解析、rollout、过滤、下采样与可视化
bash video_proxy/data/base_sources/mcq/run_pipeline.sh
```

## 输出去向

- TG 默认写入 `video_proxy/data/base_sources/tg/data/`，常见文件是 `tg_timerft_max256s_validated.jsonl` 和 `tg_tvgbench_max256s_validated.jsonl`。
- MCQ 默认写入 `video_proxy/data/base_sources/mcq/results/` 或通过 `OUTPUT_ROOT` 覆盖。
- 最终进入训练前，还需要由 `video_proxy/data/scripts/setup_base_data.sh` 采样到统一的 `$MULTI_TASK_DATA_ROOT/base/` 和 `$MULTI_TASK_DATA_ROOT/val/`。

## 注意

旧目录 `video_proxy/data/pipelines/llava_video_178k/` 和 `video_proxy/data/pipelines/temporal_grounding/` 已经迁移到这里。新代码请引用 `base_sources/` 下的路径。
