# 多任务数据混合

`mixing/` 负责把 TG、MCQ、HierSeg、Event Logic、Temporal AoT 等任务统一采样成训练入口。每种任务各自维护一个模块，`mixer.py` 负责 CLI 调度。

## 目录结构

```text
video_proxy/data/mixing/
├── common.py          # JSONL 读写、分层采样、路径处理等共享工具
├── tg.py              # Temporal Grounding
├── mcq.py             # LLaVA-Video MCQ
├── hier_seg.py        # Hierarchical Segmentation
├── event_logic.py     # Event Logic Sort
├── aot.py             # Temporal AoT
└── mixer.py           # CLI 入口：setup / mix / check
```

## 支持任务

| 模块 | problem_type | 训练来源 | 验证来源 |
| --- | --- | --- | --- |
| `tg` | `temporal_grounding` | TimeR1/TimeLens | TVGBench 采样 |
| `mcq` | `llava_mcq` | LLaVA-Video MCQ | 按 `data_source` 分层采样 |
| `hier_seg` | `L1`, `L2`, `L3_seg` | 分层标注结果 | 分层标注 val |
| `event_logic` | `event_logic_sort` | 事件关系代理数据 | 事件关系 val |
| `aot` | `seg_aot_*` | 事件进展代理数据 | 事件进展 val |

## 入口命令

所有命令都从 `train/` 根目录运行。因为 `mixer.py` 是模块入口，推荐用下面这种形式调用：

```bash
python3 -c "import sys; sys.path.insert(0, '.'); from video_proxy.data.mixing.mixer import main; main()" -- --help
```

## 步骤 1：生成 base 与 val

```bash
bash video_proxy/data/scripts/setup_base_data.sh
```

等价的 Python 调用：

```bash
python3 -c "import sys; sys.path.insert(0, '.'); from video_proxy.data.mixing.mixer import main; main()" -- \
  --data-root /path/to/VideoProxyMixed/three_task \
  setup \
  --tasks tg mcq hier_seg \
  --tg-train-source /path/to/tg_timerft_max256s_validated.jsonl \
  --tg-tvgbench-source /path/to/tg_tvgbench_max256s_validated.jsonl \
  --mcq-source /path/to/train_final_direct.jsonl \
  --hier-val-source /path/to/hier_seg_val_all.jsonl
```

典型输出：

```text
$DATA_ROOT/
├── base/
│   ├── tg_train_no_tvgbench.jsonl
│   └── mcq_train_filtered.jsonl
└── val/
    ├── tg_val_600.jsonl
    ├── mcq_val_600.jsonl
    └── hier_seg_val_150.jsonl
```

## 步骤 2：混合实验数据

```bash
python3 -c "import sys; sys.path.insert(0, '.'); from video_proxy.data.mixing.mixer import main; main()" -- \
  --data-root /path/to/VideoProxyMixed/three_task \
  mix \
  --tasks tg mcq hier_seg event_logic aot \
  --exp-name R1_f1iou \
  --hier-train /path/to/hier_seg_train_all.jsonl \
  --hier-target 5000 \
  --el-train /path/to/event_logic_train.jsonl \
  --el-target 2000 \
  --aot-train /path/to/aot_train.jsonl \
  --aot-target 10000
```

输出：

```text
$DATA_ROOT/experiments/R1_f1iou/
├── train.jsonl
└── val.jsonl
```

## 步骤 3：检查数据

```bash
python3 -c "import sys; sys.path.insert(0, '.'); from video_proxy.data.mixing.mixer import main; main()" -- \
  --data-root /path/to/VideoProxyMixed/three_task \
  check \
  --tasks tg mcq hier_seg event_logic aot
```

## 常用环境变量

| 变量 | 说明 |
| --- | --- |
| `MULTI_TASK_DATA_ROOT` | base、val、experiments 的根目录 |
| `TASKS` | 启用任务，例如 `tg mcq hier_seg` |
| `TG_TRAIN_SOURCE` | TG train JSONL 来源 |
| `TG_TVGBENCH_SOURCE` | TG val 采样来源 |
| `MCQ_SOURCE` | MCQ train JSONL 来源 |
| `HIER_TRAIN` / `HIER_VAL_SOURCE` | HierSeg train/val 来源 |
| `EL_TRAIN` / `EL_VAL_SOURCE` | Event Logic train/val 来源 |
| `AOT_TRAIN` / `AOT_VAL_SOURCE` | Temporal AoT train/val 来源 |
| `FORCE=true` | 覆盖已有 base/val 产物 |

## 新增任务

1. 在 `video_proxy/data/mixing/` 下新增任务模块。
2. 实现 `NAME`、`PROBLEM_TYPES`、`add_cli_args()`、`setup_base()`、`load_train()`、`sample_train()`、`load_val()`。
3. 在 `mixer.py` 的任务注册表中加入新模块。
4. 在 `video_proxy/training/common/multi_task_common.sh` 中补默认路径和环境变量。
