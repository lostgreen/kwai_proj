# AoT Hard QA Pipeline

## 概览

这条管线的目标是：从 seg 标注先全量构造一个 AoT raw QA 池，再用 `Qwen3-VL-8B` rollout 过滤出“能做对，但不稳定全对”的 hard cases。

当前设计遵循三条原则：
- 物理层只维护一套原视频 `2fps source frame cache`
- 逻辑 clip 一律用 frame list 表示，不把 `mp4` 当终态
- raw 阶段先全量构造，过滤后再考虑 task 配比

任务组成：
- 主盘：`action_v2t_3way`、`action_t2v_binary`
- 辅助盘：`event_v2t_3way`、`event_t2v_binary`
- 方向判断：`event_forward_reverse_binary`

## 输入与输出

输入：
- seg annotation 目录
- annotation 里可解析到的 `source_video_path`
- source frame cache 根目录 `frames_root`

中间产物：
- `action.jsonl`
- `event.jsonl`
- `event_dir.jsonl`
- `source_frame_cache/`

raw 输出：
- `action_v2t.jsonl`
- `action_t2v.jsonl`
- `event_v2t.jsonl`
- `event_t2v.jsonl`
- `event_forward_reverse.jsonl`
- `merged_raw/train.jsonl`
- `merged_raw/val.jsonl`
- `merged_raw/stats.json`

rollout 输出：
- `rollout/rollout_output.jsonl`
- `rollout/rollout_report.jsonl`
- `rollout/hard_cases.jsonl`
- `rollout/hard_cases.stats.json`

## 目录分层

物理层：
- 原视频只抽一次 `2fps` JPEG
- 所有 action/event span 都从同一套 source cache slice

逻辑层：
- 每个样本直接写成 frame-list JSONL
- `concat` 本质是拼接多个 frame list
- `reverse` 本质是对单个 frame list 做反转

这样做的好处是：
- 避免重复抽帧
- 避免为每个样本额外生成 clip mp4
- raw 数据天然和 rollout 输入格式对齐

## Stage 1: 构造 group manifests

脚本：
- [build_aot_group_manifest.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/proxy_data/youcook2_seg/temporal_aot/build_aot_group_manifest.py)

输出三类 manifest：
- `action`：一个 `event` 下的 `sub_actions`
- `event`：一个 `phase` 下的 `events`
- `event_dir`：单个 `event span`，给真正的 forward/reverse 用

最小命令：

```bash
python proxy_data/youcook2_seg/temporal_aot/build_aot_group_manifest.py \
  --annotation-dir /path/to/seg_annotations \
  --action-output data/manifests/action.jsonl \
  --event-output data/manifests/event.jsonl \
  --event-dir-output data/manifests/event_dir.jsonl \
  --complete-only \
  --filter-order \
  --min-actions 2 \
  --max-actions 8 \
  --min-events 2 \
  --max-events 8
```

## Stage 2: 构造 shared 2fps source cache

脚本：
- [hard_qa_pipeline.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py)

子命令：
- `build-source-cache`

最小命令：

```bash
python proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py build-source-cache \
  --manifest data/manifests/action.jsonl \
  --manifest data/manifests/event.jsonl \
  --manifest data/manifests/event_dir.jsonl \
  --frames-root data/source_frame_cache \
  --cache-fps 2.0
```

说明：
- 相同 source video 只会缓存一次
- dry-run 可先看预期目录与汇总

```bash
python proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py build-source-cache \
  --manifest data/manifests/action.jsonl \
  --frames-root data/source_frame_cache \
  --dry-run \
  --stats-output data/cache_plan.json
```

## Stage 3: 构造 action/event AoT raw

脚本：
- [build_aot_from_frames.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/proxy_data/youcook2_seg/temporal_aot/build_aot_from_frames.py)

输出：
- `action_v2t_3way`
- `action_t2v_binary`
- `event_v2t_3way`
- `event_t2v_binary`

最小命令：

```bash
python proxy_data/youcook2_seg/temporal_aot/build_aot_from_frames.py \
  --frames-root data/source_frame_cache \
  --action-manifest data/manifests/action.jsonl \
  --event-manifest data/manifests/event.jsonl \
  --action-v2t-output data/raw/action_v2t.jsonl \
  --action-t2v-output data/raw/action_t2v.jsonl \
  --event-v2t-output data/raw/event_v2t.jsonl \
  --event-t2v-output data/raw/event_t2v.jsonl \
  --action-t2v-max-duration 90 \
  --event-t2v-max-duration 60
```

这里的默认取舍是：
- action-level 是主盘，所以 `v2t=3way`、`t2v=binary`
- event-level 做辅助，`t2v` 额外控时长，避免超帧预算

## Stage 4: 构造真正的 event forward/reverse

脚本：
- [build_event_forward_reverse_from_frames.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/proxy_data/youcook2_seg/temporal_aot/build_event_forward_reverse_from_frames.py)

最小命令：

```bash
python proxy_data/youcook2_seg/temporal_aot/build_event_forward_reverse_from_frames.py \
  --event-manifest data/manifests/event_dir.jsonl \
  --frames-root data/source_frame_cache \
  --output data/raw/event_forward_reverse.jsonl \
  --sample-mode one_per_event
```

这里的 reverse 不是旧 sort 数据恢复出来的“顺序反了”，而是：
- 对同一个 `event span` 的 frame list 做真实倒放

## Stage 5: 合并 raw pool

子命令：
- `merge-raw`

最小命令：

```bash
python proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py merge-raw \
  --input data/raw/action_v2t.jsonl \
  --input data/raw/action_t2v.jsonl \
  --input data/raw/event_v2t.jsonl \
  --input data/raw/event_t2v.jsonl \
  --input data/raw/event_forward_reverse.jsonl \
  --output-dir data/merged_raw \
  --val-ratio 0.1
```

输出：
- `train.jsonl`
- `val.jsonl`
- `stats.json`

`stats.json` 会给出：
- 各 `problem_type` 数量
- `domain_l1 / domain_l2` 分布
- 平均帧数
- 平均时长

## Stage 6: rollout-filter hard cases

子命令：
- `rollout-filter`

默认目标：
- 模型至少有一次答对
- 但不能稳定全对
- 总体落在 hard-but-solvable 区间

默认参数：
- `model_path=/m2v_intern/xuboshen/models/Qwen3-VL-8B-Instruct`
- `reward_function=verl/reward_function/mixed_proxy_reward.py:compute_score`
- `num_rollouts=8`
- `min_mean_reward=0.125`
- `max_mean_reward=0.625`
- `min_success_count=1`
- `success_threshold=1.0`
- `target_total=5000`

最小命令：

```bash
python proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py rollout-filter \
  --input data/merged_raw/train.jsonl \
  --output-dir data/rollout
```

它会顺序编排：
1. [local_scripts/offline_rollout_filter.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/local_scripts/offline_rollout_filter.py)
2. [filter_rollout_hard_cases.py](/Users/lostgreen/Desktop/Codes/VideoProxy/train/proxy_data/youcook2_seg/temporal_aot/filter_rollout_hard_cases.py)

最终输出：
- `rollout_output.jsonl`
  rollout 阶段的直接保留结果
- `rollout_report.jsonl`
  每条样本的 reward 明细
- `hard_cases.jsonl`
  最终 hard-case 训练集
- `hard_cases.stats.json`
  最终保留分布统计

如果只想先看命令和路径，不实际跑模型：

```bash
python proxy_data/youcook2_seg/temporal_aot/hard_qa_pipeline.py rollout-filter \
  --input data/merged_raw/train.jsonl \
  --output-dir data/rollout \
  --dry-run \
  --stats-output data/rollout_plan.json
```

## 规模预估

按你现在这个问题设定，`10k videos` 下更现实的 raw 规模大概是：
- `100k ~ 180k` raw QA

更细一点：
- `action_v2t` 和 `action_t2v` 通常会占主盘
- `event_t2v` 因为时长约束会更少
- `event_forward_reverse` 往往也能提供几十 k 候选

这只是经验带，不是硬上限。真正数量还会受：
- 标注完整度
- `_order_distinguishable`
- 时长过滤
- 去重
- rollout 后 hard-case 收缩

## 常见检查点

先只看规划，不落盘：
- `build-source-cache --dry-run`
- `rollout-filter --dry-run`

先看 raw 分布：
- `data/merged_raw/stats.json`

再看 hard-case 分布：
- `data/rollout/hard_cases.stats.json`

如果出现 hard cases 过少，优先检查：
- `min_mean_reward / max_mean_reward` 是否过窄
- `min_success_count` 是否过严
- `success_threshold` 是否和 reward 语义一致

## 一句话主线

`manifest -> source cache -> frame-list raw tasks -> merge raw -> rollout-filter`

这就是当前 AoT hard-QA 大盘构造的主入口。
