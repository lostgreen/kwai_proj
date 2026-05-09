# 数据脚本入口

这里放跨模块的一键脚本和轻量转换工具。具体业务逻辑仍然应该留在 `base_sources/`、`pipelines/` 或 `mixing/` 中。

## 常用命令

从 `train/` 根目录运行：

```bash
# 采样并生成统一 base/ 与 val/ 数据
bash video_proxy/data/scripts/setup_base_data.sh

# 刷新 TG base，并可接入 TimeLens rollout 结果
bash video_proxy/data/scripts/refresh_tg_base_with_timelens.sh

# 刷新 MCQ base，使用 LLaVA shard rollout 结果筛选
bash video_proxy/data/scripts/refresh_mcq_base_with_llava_shards.sh

# 为基础数据准备离线帧缓存
bash video_proxy/data/scripts/prepare_base_offline_frames.sh

# 查看 JSONL CoT 转换工具参数
python video_proxy/data/scripts/convert_jsonl_to_cot.py --help
```

## 主要脚本

- `setup_base_data.sh`：把已准备好的 TG、MCQ、HierSeg、EventLogic、AoT 等数据采样到 `$MULTI_TASK_DATA_ROOT/base/` 和 `$MULTI_TASK_DATA_ROOT/val/`。
- `refresh_tg_base_with_timelens.sh`：刷新 TG 基座数据，合并 TimeR1 与 TimeLens。
- `refresh_mcq_base_with_llava_shards.sh`：从 MCQ rollout shards 中筛选并刷新 MCQ 基座数据。
- `prepare_base_offline_frames.sh`：为训练数据准备可复用的离线帧。
- `convert_jsonl_to_cot.py`：把 JSONL prompt 转成 CoT 训练格式。

## 输出

脚本通常不会把大数据写进仓库，而是写到外部数据根目录，例如：

```text
$MULTI_TASK_DATA_ROOT/
├── base/
├── val/
└── experiments/
```

路径默认值来自 `video_proxy/training/common/multi_task_common.sh`，可以通过环境变量覆盖。
