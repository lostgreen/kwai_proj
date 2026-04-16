# Multi-Task Data Management

模块化多任务训练数据管理。每种数据类型有独立模块，通过 `mixer.py` 统一调度。

## 目录结构

```
local_scripts/data/
├── common.py          # 共享工具: load_jsonl, write_jsonl, stratified_sample 等
├── tg.py              # Temporal Grounding (TimeRFT train + TVGBench val)
├── mcq.py             # LLaVA Video MCQ
├── hier_seg.py        # Hierarchical Segmentation (L1/L2/L3_seg)
├── event_logic.py     # Event Logic Sort (event_logic_sort)
└── mixer.py           # CLI 入口: setup / mix / check
```

## 数据类型

| 模块 | problem_type | Train | Val 采样 |
|------|-------------|-------|---------|
| `tg` | `temporal_grounding` | TimeRFT 全量 (~2.2k) | TVGBench random sample |
| `mcq` | `llava_mcq` | 全量 | 按 `data_source` 分层 |
| `hier_seg` | `L1`, `L2`, `L3_seg` | 按 problem_type 等比例到 target | 按 problem_type 等比例 |
| `event_logic` | `event_logic_sort` | 按 problem_type 等比例到 target | 按 problem_type 分层 |

## 使用方法

所有命令在 **train/** 目录下执行。

### Step 1: 生成基座数据 (只需运行一次)

```bash
# 使用默认路径 (由 three_task_common.sh 配置)
bash local_scripts/setup_base_data.sh

# 或者直接调用 Python
python3 -c "
import sys; sys.path.insert(0, '.')
from local_scripts.data.mixer import main; main()
" -- \
    --data-root /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/three_task \
    --tasks tg mcq hier_seg \
    setup \
    --tg-timerft-json /path/to/train_2k5.json \
    --tg-tvgbench-json /path/to/tvgbench.json \
    --tg-video-base /path/to/TimeR1-Dataset \
    --mcq-source proxy_data/llava_video_178k/results/train_final.jsonl \
    --hier-val-source /path/to/val_all.jsonl
```

生成目录:
```
$DATA_ROOT/
├── base/
│   ├── tg_train_no_tvgbench.jsonl
│   └── mcq_train_filtered.jsonl
└── val/
    ├── tvgbench_val_150.jsonl
    ├── mcq_val_150.jsonl
    └── hier_seg_val_150.jsonl
```

### Step 2: 混合实验数据

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from local_scripts.data.mixer import main; main()
" -- \
    --data-root /m2v_intern/.../three_task \
    --tasks tg mcq hier_seg \
    mix \
    --exp-name R1_f1iou \
    --hier-train /path/to/train_all.jsonl \
    --hier-target 5000
```

生成:
```
$DATA_ROOT/experiments/R1_f1iou/
├── train.jsonl    # TG + MCQ + HierSeg(5k) 混合
└── val.jsonl      # 各任务 val 合并
```

### Step 3: 启动训练

```bash
# 直接运行 (2卡默认)
bash local_scripts/run_multi_task.sh

# 消融实验
bash local_scripts/hier_seg_ablations/reward_ablation/exp_r1_f1iou.sh
```

### 检查数据完整性

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from local_scripts.data.mixer import main; main()
" -- \
    --data-root /m2v_intern/.../three_task \
    --tasks tg mcq hier_seg \
    check
```

## 添加新任务

1. 在 `local_scripts/data/` 下创建 `new_task.py`
2. 实现: `NAME`, `PROBLEM_TYPES`, `add_cli_args()`, `setup_base()`, `load_train()`, `sample_train()`, `load_val()`
3. 在 `mixer.py` 的 `_ALL_MODULES` 中注册
4. 在 `three_task_common.sh` 中添加对应的环境变量

## 环境变量 (可在 shell 中覆盖)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `THREE_TASK_DATA_ROOT` | `/m2v_intern/.../three_task` | 数据根目录 |
| `TASKS` | `tg mcq hier_seg` | 启用的任务列表 |
| `HIER_TRAIN` | `.../train_all.jsonl` | Hier Seg 训练数据源 |
| `HIER_TARGET` | `5000` | Hier Seg 训练采样目标 |
| `EL_TRAIN` | (空) | Event Logic 训练数据源 |
| `EL_TARGET` | `2000` | Event Logic 训练采样目标 |
| `FORCE` | `false` | 强制重新生成 |
