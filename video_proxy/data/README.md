# VideoProxy 数据目录

这个目录只放数据构建、筛选、混合和格式转换相关代码。当前推荐把“基础数据源”和“代理任务构造”分开管理：

- `base_sources/`：外部基础数据集的清洗与刷新，例如 TG、MCQ。
- `pipelines/`：VideoProxy 自己的候选筛选、分层标注、事件边界、事件进展和事件关系构造。
- `mixing/`：把各任务的 base/val/experiment 数据统一混合成训练入口。
- `scripts/`：常用一键脚本，薄封装 `base_sources/`、`pipelines/` 和 `mixing/`。

## 常用入口

从仓库的 `train/` 根目录运行：

```bash
# 生成/刷新多任务 base 与 val 数据，依赖各基础数据源已经准备好
bash video_proxy/data/scripts/setup_base_data.sh

# 刷新 Temporal Grounding base，含 TimeR1 与 TimeLens rollout 合并
bash video_proxy/data/scripts/refresh_tg_base_with_timelens.sh

# 刷新 LLaVA-Video MCQ base，含 rollout shard 选择
bash video_proxy/data/scripts/refresh_mcq_base_with_llava_shards.sh

# 准备离线帧缓存
bash video_proxy/data/scripts/prepare_base_offline_frames.sh
```

## 推荐工作流

1. 在 `base_sources/` 中把外部数据集转成统一 JSONL。
2. 在 `pipelines/data_curation/` 中筛选可用于标注或代理任务的视频候选。
3. 在 `pipelines/proxy_construction/` 中构造层级分割、事件边界、事件进展、事件关系等代理数据。
4. 在 `mixing/` 中把基础任务和代理任务混成训练用的 `train.jsonl` / `val.jsonl`。

## 产物约定

- 代码内默认路径多指向集群数据盘；本地仓库通常不保存大体量视频、帧缓存或 rollout 产物。
- 可复用的中间数据建议放在外部数据根目录下，例如 `$MULTI_TASK_DATA_ROOT/base/`、`$MULTI_TASK_DATA_ROOT/val/`、`$MULTI_TASK_DATA_ROOT/experiments/<name>/`。
- 新增脚本时优先放到对应功能目录；只有跨模块的一键入口才放进 `scripts/`。
