# Event Boundary：事件边界数据构造

本目录从层级标注 JSON 构造 event-boundary proxy，也就是论文叙事中的边界感知任务。它覆盖 `L1`、`L2`、`L3_seg` 三类层级边界标注。

## 文件说明

- `build_boundary_frame_list_data.py`：主构造脚本，从 annotation JSON 直接生成 EasyR1 JSONL。
- `boundary_prompts.py`：L1/L2/L3_seg 的 prompt 模板。

## 运行命令

三层全部构建：

```bash
python video_proxy/data/pipelines/proxy_construction/event_boundary/build_boundary_frame_list_data.py \
  --annotation-dir /path/to/annotations \
  --output-dir /path/to/event_boundary \
  --levels L1 L2 L3_seg \
  --complete-only
```

限制规模并做领域均衡：

```bash
python video_proxy/data/pipelines/proxy_construction/event_boundary/build_boundary_frame_list_data.py \
  --annotation-dir /path/to/annotations \
  --output-dir /path/to/event_boundary_small \
  --levels L1 L2 L3_seg \
  --train-per-level 1000 \
  --total-val 100 \
  --balance-per-level 1200 \
  --complete-only
```

只构建 L2：

```bash
python video_proxy/data/pipelines/proxy_construction/event_boundary/build_boundary_frame_list_data.py \
  --annotation-dir /path/to/annotations \
  --output-dir /path/to/event_boundary_l2 \
  --levels L2 \
  --l2-mode phase \
  --complete-only
```

## 输出

`--output-dir` 下会生成 `train.jsonl`、`val.jsonl` 或按层级拆分的 JSONL。每条记录包含 `messages`、`prompt`、`answer`、`videos`、`problem_type`、`metadata` 等字段。

## 注意

当前脚本仍兼容旧 mp4 clip 输入；新数据建议优先走 frame-list 表示。旧 clip 物化脚本只作为本地历史材料保留，不属于主提交路径。
