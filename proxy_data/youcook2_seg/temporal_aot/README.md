# Temporal AoT

当前推荐流程已经切到“`2fps source frame cache + frame-list JSONL`”这条线，不再把 `mp4 clip` 当成主终态。

完整操作手册见：
- [docs/aot/hard_qa_pipeline.md](/Users/lostgreen/Desktop/Codes/VideoProxy/train/docs/aot/hard_qa_pipeline.md)

## 当前推荐管线

目标是从 seg 标注直接产出大规模 AoT raw pool，再用 rollout 过滤出“模型能做对，但不稳定全对”的 hard QA。

主盘：
- `action_v2t_3way`
- `action_t2v_binary`

辅助盘：
- `event_v2t_3way`
- `event_t2v_binary`
- `event_forward_reverse_binary`

核心特点：
- 物理层只维护一套原视频 `2fps` frame cache
- 逻辑 clip 全部用 frame list 表示
- `forward/reverse` 直接对单个 `event span` 的 frame list 做真实倒放
- raw 阶段先全量构造，不预先控配比
- rollout 阶段再筛 hard-but-solvable 样本

## 主要脚本

- [build_aot_group_manifest.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/proxy_data/youcook2_seg/temporal_aot/build_aot_group_manifest.py)
  从 seg annotation 生成 `action / event / event_dir` manifest
- [hard_qa_pipeline.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py)
  编排 `build-source-cache / merge-raw / rollout-filter`
- [build_aot_from_frames.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/proxy_data/youcook2_seg/temporal_aot/build_aot_from_frames.py)
  从 manifest + source cache 产出 `action/event v2t/t2v`
- [build_event_forward_reverse_from_frames.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/proxy_data/youcook2_seg/temporal_aot/build_event_forward_reverse_from_frames.py)
  从 `event_dir` manifest 产出真实倒放判断任务
- [filter_rollout_hard_cases.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/proxy_data/youcook2_seg/temporal_aot/filter_rollout_hard_cases.py)
  从 rollout report 里筛 `hard_cases.jsonl`

## 最小流程

```bash
cd /Users/lostgreen/Desktop/Codes/VideoProxy/train

# 1) build manifests
python proxy_data/youcook2_seg/temporal_aot/build_aot_group_manifest.py \
  --annotation-dir /path/to/seg_annotations \
  --action-output data/manifests/action.jsonl \
  --event-output data/manifests/event.jsonl \
  --event-dir-output data/manifests/event_dir.jsonl \
  --complete-only \
  --filter-order

# 2) build shared 2fps source cache
python proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py build-source-cache \
  --manifest data/manifests/action.jsonl \
  --manifest data/manifests/event.jsonl \
  --manifest data/manifests/event_dir.jsonl \
  --frames-root data/source_frame_cache

# 3) build action/event AoT tasks
python proxy_data/youcook2_seg/temporal_aot/build_aot_from_frames.py \
  --frames-root data/source_frame_cache \
  --action-manifest data/manifests/action.jsonl \
  --event-manifest data/manifests/event.jsonl \
  --action-v2t-output data/raw/action_v2t.jsonl \
  --action-t2v-output data/raw/action_t2v.jsonl \
  --event-v2t-output data/raw/event_v2t.jsonl \
  --event-t2v-output data/raw/event_t2v.jsonl

# 4) build event forward/reverse
python proxy_data/youcook2_seg/temporal_aot/build_event_forward_reverse_from_frames.py \
  --event-manifest data/manifests/event_dir.jsonl \
  --frames-root data/source_frame_cache \
  --output data/raw/event_forward_reverse.jsonl

# 5) merge raw pool
python proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py merge-raw \
  --input data/raw/action_v2t.jsonl \
  --input data/raw/action_t2v.jsonl \
  --input data/raw/event_v2t.jsonl \
  --input data/raw/event_t2v.jsonl \
  --input data/raw/event_forward_reverse.jsonl \
  --output-dir data/merged_raw

# 6) rollout-filter hard cases
python proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py rollout-filter \
  --input data/merged_raw/train.jsonl \
  --output-dir data/rollout
```

## rollout-filter 默认值

- `model_path=/m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct`
- `num_rollouts=8`
- `min_mean_reward=0.125`
- `max_mean_reward=0.625`
- `min_success_count=1`
- `target_total=5000`

默认输出：
- `rollout_output.jsonl`
- `rollout_report.jsonl`
- `hard_cases.jsonl`
- `hard_cases.stats.json`

## 规模预估

如果你的 seg 数据量级在 `10k videos` 左右，这条线比较现实的 raw 规模大概会落在：
- `100k ~ 180k` raw QA

其中：
- action-level 通常会占大头
- `event_t2v` 会因为时长过滤少一些
- `event_forward_reverse` 往往也能贡献一批几十 k 量级候选

## 旧版脚本说明

[build_aot_from_seg.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/proxy_data/youcook2_seg/temporal_aot/build_aot_from_seg.py) 还保留在目录里，适合小规模直接从 seg 标注产 AoT 的旧路径或对照实验，但它不再是当前 hard-QA 大盘构造的主入口。
