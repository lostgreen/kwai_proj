# TimeLens-100K 数据源

## 基本信息

- **来源**: TimeLens-100K
- **路径**: `/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/`
- **规模**: 19,466 条
- **标注文件**: `timelens-100k.jsonl`
- **视频目录**: `video_shards/{source}/{video_id}.mp4`

## 数据格式

```json
{
  "source": "cosmo_cap",
  "video_path": "cosmo_cap/BVs52yd-RUQ.mp4",
  "duration": 117.4,
  "events": [
    {
      "query": "When does the speaker introduce himself and the company?",
      "span": [[0.0, 5.0]]
    },
    {
      "query": "Show me a close-up shot of the macadamia nut oil bottle.",
      "span": [[8.0, 12.0]]
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `source` | str | 来源域（7 种） |
| `video_path` | str | 相对于 `video_shards/` 的视频路径 |
| `duration` | float | 视频时长（秒） |
| `events` | list | 时序事件列表，每个含 `query`（事件描述）和 `span`（[[start, end]]） |

### 与 ET-Instruct 格式对比

| 维度 | ET-Instruct | TimeLens |
|------|-------------|---------|
| 时间戳 | `tgt` 成对列表 | `events[i].span` |
| 事件描述 | GPT 回复文本中提取 | `events[i].query` |
| 时长字段 | `duration` ✓ | `duration` ✓ |
| 已有 L2 结构 | 是（slc/dvc task） | 是（query+span） |

## 域分布与特点

| Domain | 数量 | 特点 | L2 标注适合度 |
|--------|------|------|--------------|
| cosmo_cap | 9,549 | 产品演示/YouTube 教程 | ★★★★ |
| internvid_vtime | 4,651 | InternVid 时序标注，开放域 | ★★★ |
| didemo | 2,955 | DiDeMo 短视频检索 | ★★ |
| queryd | 1,518 | QuerYD 时序定位 | ★★ |
| **hirest_step** | **387** | **HiREST 层次步骤标注** | **★★★★★** |
| hirest_grounding | 249 | HiREST grounding | ★★★★ |
| hirest | 157 | HiREST 基础 | ★★★★ |

**hirest 系列为最高优先级**：HiREST (Hierarchical Retrieval with Steps) 本身就是层次结构标注，steps 字段天然对应 L2 事件。

## 统计摘要（原始数据）

- 总条数: 19,466
- 时长: 5.2s–498.9s，均值 106.3s
- **60-240s 范围**: 13,568 条 (70%)
- **events ≥5**: 18,981 条 (97%)
- events 均值: 5.0

预估筛选后约 **12,000-13,000** 个独立视频。

## 筛选进度

- [x] 数据格式探索
- [ ] text_filter.py 运行
- [ ] Stage A: Route D (VLM-Curated 物理过程审查)
- [ ] VLM Vision Filter (6 帧视觉校验)
- [ ] 可视化验证
- [ ] 与 ET-Instruct candidates 合并

## 运行指令

> **所有命令均从 `train/` (EasyR1) 目录执行**，使用相对路径。

### Step 0: 文本筛选

```bash
# Dry run（仅看统计，不写文件）
python proxy_data/data_curation/timelens_100k/text_filter.py \
    --input /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/timelens-100k.jsonl \
    --config proxy_data/data_curation/configs/timelens_100k.yaml \
    --dry-run

# 正式筛选
python proxy_data/data_curation/timelens_100k/text_filter.py \
    --input /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/timelens-100k.jsonl \
    --output proxy_data/data_curation/results/timelens_100k/passed_timelens.jsonl \
    --config proxy_data/data_curation/configs/timelens_100k.yaml
```

**产出**：
- `proxy_data/data_curation/results/timelens_100k/passed_timelens.jsonl`
- `proxy_data/data_curation/results/timelens_100k/filter_summary.json`

### Stage A + Vision Filter: 一键 Pipeline

```bash
# 抽样试跑 (Route D 200 条)
bash proxy_data/data_curation/timelens_100k/run_pipeline.sh --sample

# 全量: Stage A (Route D) → Vision Filter → 最终候选
VIDEO_ROOT=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/video_shards \
    bash proxy_data/data_curation/timelens_100k/run_pipeline.sh --full

# Stage A 已完成，只跑 Vision Filter
bash proxy_data/data_curation/timelens_100k/run_pipeline.sh --vision-only
```

> `run_pipeline.sh` 内部会自动 `cd` 到脚本所在目录，因此从 `train/` 直接运行即可。

**产出**（均在 `proxy_data/data_curation/results/timelens_100k/` 下）：
- `stage_a_results_keep.jsonl` — Stage A (Route D) 通过
- `vision_results_keep.jsonl` — **最终候选**（VLM 视觉校验通过）

### 单步运行 Stage A (Route D)

```bash
# 抽样 200 条看分布
python proxy_data/data_curation/timelens_100k/stage_a_coarse_filter.py \
    --input proxy_data/data_curation/results/timelens_100k/passed_timelens.jsonl \
    --output proxy_data/data_curation/results/timelens_100k/stage_a_results.jsonl \
    --sample-n 200

# 全量评估（断点续评）
python proxy_data/data_curation/timelens_100k/stage_a_coarse_filter.py \
    --input proxy_data/data_curation/results/timelens_100k/passed_timelens.jsonl \
    --output proxy_data/data_curation/results/timelens_100k/stage_a_results.jsonl \
    --no-sample --resume --workers 16
```

### 单步运行 VLM Vision Filter

```bash
python proxy_data/data_curation/shared/vision_filter.py \
    --input proxy_data/data_curation/results/timelens_100k/stage_a_results_keep.jsonl \
    --output proxy_data/data_curation/results/timelens_100k/vision_results.jsonl \
    --video-root /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/video_shards \
    --video-field video_path \
    --workers 4
```

### 可视化验证（帧 + 时间线）

```bash
# 转换 keep 样本为 segmentation_visualize 格式（含 1fps 抽帧）
# --output 建议指向数据集目录，避免大量帧文件进入 git
python proxy_data/data_curation/shared/convert_to_viz.py \
    --input proxy_data/data_curation/results/timelens_100k/stage_a_results_keep.jsonl \
    --output /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/viz_candidates/ \
    --data-source timelens \
    --video-root /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/video_shards/ \
    --workers 8

# 仅生成 JSON（不抽帧）
# python proxy_data/data_curation/shared/convert_to_viz.py \
#     --input ... --output ... --data-source timelens --no-frames

# 启动可视化服务器
python data_visualization/segmentation_visualize/server.py \
    --annotation-dir /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeLens-100K/viz_candidates/ \
    --port 8765

# 浏览器打开 http://127.0.0.1:8765
```
