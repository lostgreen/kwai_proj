# local_scripts — 训练脚本与数据管理

## 当前实验: Hier Seg Reward 消融

多任务混合训练 (TG + MCQ + Hier Seg)，对比不同 reward function 对 Hier Seg 的影响。

### 快速开始

```bash
cd train/   # 在 repo 的 train/ 目录下执行

# 1. 生成基座数据 (首次运行一次)
bash local_scripts/setup_base_data.sh

# 2. 运行 R1 baseline (F1-IoU reward)
bash local_scripts/hier_seg_ablations/reward_ablation/exp_r1_f1iou.sh

# 3. 批量运行所有消融
bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh
```

### 消融实验

| 实验 | Reward | 范围 | 脚本 |
|------|--------|------|------|
| R1 | F1-IoU (baseline) | [0, 1] | `exp_r1_f1iou.sh` |
| R3 | DP-F1 + Instance Count | [0, 2] | `exp_r3_dp_f1.sh` |
| R4 | Segment Matching | [0, 1] | `exp_r4_seg_match.sh` |

### 配置参数

通过环境变量覆盖，在运行脚本前 export:

**任务控制:**
```bash
export TASKS="tg mcq hier_seg"     # 启用的任务 (空格分隔)
export HIER_TARGET=5000             # Hier Seg 训练数据量
export EL_TARGET=2000               # Event Logic 训练数据量 (需加入 TASKS)
```

**硬件:**
```bash
# 2卡 (默认)
bash exp_r1_f1iou.sh

# 8卡
N_GPUS_PER_NODE=8 ROLLOUT_BS=16 GLOBAL_BS=16 bash exp_r1_f1iou.sh
```

**快速调试:**
```bash
MAX_STEPS=30 bash exp_r1_f1iou.sh
```

**训练超参:**
```bash
export LR=5e-7                  # 学习率
export KL_COEF=0.04             # KL loss 系数
export FILTER_LOW=0.2           # Online filtering 下界
export FILTER_HIGH=0.8          # Online filtering 上界
export MAX_RESPONSE_LEN=1024    # 最大生成长度
```

### 数据流

```
proxy_data/ pipelines          local_scripts/data/           服务器数据目录
(生成原始数据)                  (采样 + 混合)                 (训练读取)
                                                             
run_pipeline.sh (TG)    ──→    tg.py: copy train             $DATA_ROOT/
prepare_mcq.py (MCQ)    ──→    mcq.py: copy train + val      ├── base/
build_hier_data.py      ──→    hier_seg.py: sample val        ├── val/
event_logic pipeline    ──→    event_logic.py: sample val     └── experiments/
                               mixer.py: setup / mix / check       └── {EXP_NAME}/
                                    ↓                                   ├── train.jsonl
                               setup_base_data.sh (一次性)              └── val.jsonl
                               run_multi_task.sh (每次训练)
```

### 目录结构

```
local_scripts/
├── multi_task_common.sh        # 共用配置 (模型/硬件/算法/数据路径)
├── run_multi_task.sh           # 训练入口 (数据混合 + GPU filler + 训练)
├── setup_base_data.sh          # 一键生成 base/ + val/ (只需一次)
├── gpu_filler.py               # GPU 利用率保持工具
│
├── data/                       # 模块化数据管理
│   ├── common.py               #   共享工具函数
│   ├── tg.py                   #   Temporal Grounding
│   ├── mcq.py                  #   LLaVA Video MCQ
│   ├── hier_seg.py             #   Hierarchical Segmentation
│   ├── event_logic.py          #   Event Logic Sort
│   ├── mixer.py                #   CLI: setup/mix/check
│   └── README.md               #   详细使用文档
│
├── hier_seg_ablations/         # Hier Seg 消融实验
│   ├── reward_ablation/        #   Reward 消融 (R1/R3/R4)
│   ├── prompt_ablation/        #   Prompt 消融 (V4)
│   ├── build_hier_data.py      #   → proxy_data/ symlink
│   └── eval_baseline_rollout.py
│
├── event_logic_ablations/      # Event Logic 消融
├── aot_ablations/              # AOT 消融
│
├── filter_bad_videos.py        # 视频有效性检查
├── offline_rollout_filter.py   # Rollout 过滤
└── sample_rollout_analysis.py  # Rollout 分析
```
