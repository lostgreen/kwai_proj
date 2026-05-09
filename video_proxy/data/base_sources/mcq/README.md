# LLaVA-Video MCQ 基础数据

这里管理 LLaVA-Video-178K MCQ 的基础数据构建。当前功能分为三块：数据准备、rollout 选择、可视化筛选。

## 目录结构

- `prepare/prepare_mcq.py`：把原始 MCQ JSON 解析成统一 JSONL。
- `prepare/convert_to_direct.py`：把 MCQ 样本转成 direct-answer 格式。
- `rollout/reward.py`：MCQ rollout 的奖励函数。
- `rollout/sample_pilot.py`：按来源和时长分层采样 pilot 集。
- `rollout/select_from_shards.py`：从多个 rollout shard 中按准确率筛选样本。
- `rollout/resume_helper.py`、`rollout/recover_report.py`：处理断点续跑和报告恢复。
- `visualization/filter_and_downsample.py`：按准确率过滤并下采样。
- `visualization/visualize.py`、`visualization/plot_source_pie.py`：画分布图和来源占比图。
- `check_format.py`：检查最终 JSONL 的字段和 prompt 格式。
- `run_pipeline.sh`：一键入口。

## 一键运行

从 `train/` 根目录运行：

```bash
DATASET_ROOT=/path/to/LLaVA-Video-178K \
MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
OUTPUT_ROOT=/path/to/mcq_results \
NUM_GPUS=8 \
NUM_ROLLOUTS=8 \
ROLLOUT_SCOPE=full \
bash video_proxy/data/base_sources/mcq/run_pipeline.sh
```

常用参数：

- `ROLLOUT_SCOPE=pilot`：先分层采样 pilot，再 rollout。
- `ROLLOUT_SCOPE=full`：直接对全量 MCQ 做 rollout。
- `MIN_ACC` / `MAX_ACC`：控制保留样本的平均准确率区间。
- `TARGET_TOTAL=0`：不过采样，保留过滤后的全部样本。
- `FORCE=1`：忽略已有产物，强制重跑。

## 分步运行

```bash
# 1. 原始数据转统一 JSONL
python video_proxy/data/base_sources/mcq/prepare/prepare_mcq.py \
  --dataset-root /path/to/LLaVA-Video-178K \
  --output /path/to/mcq_all.jsonl

# 2. pilot 分层采样
python video_proxy/data/base_sources/mcq/rollout/sample_pilot.py \
  --input /path/to/mcq_all.jsonl \
  --output /path/to/pilot_sample.jsonl \
  --per-cell 5000 \
  --seed 42

# 3. 从 rollout shards 中筛选
python video_proxy/data/base_sources/mcq/rollout/select_from_shards.py \
  --input-dir /path/to/rollout_shards \
  --output /path/to/train_final.jsonl \
  --min-acc 0.0 \
  --max-acc 0.375

# 4. 转 direct-answer 格式
python video_proxy/data/base_sources/mcq/prepare/convert_to_direct.py \
  --input /path/to/train_final.jsonl \
  --output /path/to/train_final_direct.jsonl

# 5. 检查格式
python video_proxy/data/base_sources/mcq/check_format.py \
  /path/to/train_final_direct.jsonl
```

## 输出

常见产物包括：

- `mcq_all.jsonl`：解析后的全量 MCQ。
- `pilot_sample.jsonl`：pilot 模式下的分层采样集。
- `rollout_kept.jsonl` / `rollout_report.jsonl`：rollout 后保留样本和报告。
- `train_final.jsonl` / `train_final_direct.jsonl`：进入 base 数据混合的最终 MCQ 文件。
- `*.png` / `*.html`：可视化筛选报告。

## 注意

旧路径 `video_proxy/data/pipelines/llava_video_178k/` 已迁移到本目录。新脚本和 README 请使用 `video_proxy/data/base_sources/mcq/`。
