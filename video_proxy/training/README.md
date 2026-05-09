# VideoProxy Training

训练入口按模型划分，只保留可复用的 teacher 训练和 OPD 训练脚本。EasyR1/verl 框架代码仍保持在仓库原位置，`video_proxy/training/launchers/run_multi_task.sh` 作为公共底层执行器。

## Model Entrypoints

每个模型目录都有两个脚本：

- `teacher_train_ema_grpo.sh`: 训练 teacher，用 EMA-GRPO。
- `opd_train.sh`: 用已有 teacher checkpoint 做 OPD 训练。

```
video_proxy/training/models/
├── qwen3_vl_4b/
│   ├── teacher_train_ema_grpo.sh
│   └── opd_train.sh
├── qwen3_vl_8b/
│   ├── teacher_train_ema_grpo.sh
│   └── opd_train.sh
├── qwen2_5_vl_3b/
│   ├── teacher_train_ema_grpo.sh
│   └── opd_train.sh
└── qwen2_5_vl_7b/
    ├── teacher_train_ema_grpo.sh
    └── opd_train.sh
```

## Quick Start

```bash
cd train/

# Qwen3-VL-4B teacher
bash video_proxy/training/models/qwen3_vl_4b/teacher_train_ema_grpo.sh

# Qwen3-VL-4B OPD
TEACHER_MODEL_PATH=/path/to/teacher \
  bash video_proxy/training/models/qwen3_vl_4b/opd_train.sh

# Qwen2.5-VL-7B teacher
bash video_proxy/training/models/qwen2_5_vl_7b/teacher_train_ema_grpo.sh
```

## Common Overrides

```bash
EXP_NAME=my_run \
TASKS="tg mcq hier_seg" \
N_GPUS_PER_NODE=8 \
ROLLOUT_BS=32 \
GLOBAL_BS=32 \
bash video_proxy/training/models/qwen3_vl_8b/teacher_train_ema_grpo.sh
```

OPD 默认是 single-teacher；如果要用 multi-teacher，设置 `AOT_TEACHER_MODEL_PATH`、`SEG_TEACHER_MODEL_PATH`、`EVENTLOGIC_TEACHER_MODEL_PATH` 并按需设置 `OPD_TEACHER_KEY`。

## Directory Roles

```
video_proxy/training/
├── models/       # 面向使用者的训练入口，按模型划分
├── recipes/      # teacher/opd 公共训练 recipe
├── launchers/    # 底层 runner 和兼容入口
├── common/       # 共用环境变量、数据和硬件默认值
├── tools/        # GPU filler、rollout 过滤、checkpoint 工具
└── debug/        # 训练调试脚本
```
