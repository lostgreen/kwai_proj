# Temporal Grounding 消融实验

基于 Time-R1 的 TimeRFT 训练数据，在 VideoProxy 256 frames / 2fps 框架下进行 temporal grounding RL 消融。

## 实验矩阵

| Exp | 名称 | CoT | Response 长度 | 说明 |
|-----|------|-----|---------------|------|
| 1 | `exp1_no_cot` | ✗ | 256 | 直接输出时间段 |
| 2 | `exp2_cot` | ✓ | 1024 | 先 `<think>` 分析再输出 |

## 数据

- **训练集**: TimeRFT 2.5K → 过滤 >256s → **2148 条**
- **验证集**: TVGBench 800 条（全部 ≤256s）
- **视频路径**: `/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset/timerft_data/`

## Reward

`temporal_grounding` reward (IoU × 归一化距离惩罚，参考 Time-R1 iou_v2):

```
reward = IoU(pred, gt) × (1 - |Δs_norm|) × (1 - |Δe_norm|)
```

其中 `Δs_norm = (gt_s - pred_s) / duration`，对 start/end 偏移做归一化惩罚。

## 运行

```bash
cd /path/to/train

# 实验 1: No CoT
bash local_scripts/tg_ablations/exp1_no_cot.sh

# 实验 2: CoT
bash local_scripts/tg_ablations/exp2_cot.sh
```

## 文件结构

```
local_scripts/tg_ablations/
├── common.sh          # 共用超参数
├── launch_train.sh    # 统一训练入口
├── exp1_no_cot.sh     # 实验 1: 无 CoT
├── exp2_cot.sh        # 实验 2: 有 CoT
└── README.md          # 本文件

proxy_data/temporal_grounding/
├── DESIGN.md          # 设计文档
├── build_dataset.py   # 数据转换脚本
└── data/
    ├── timerft_train_max256s_easyr1.jsonl      # No CoT 训练集 (2148)
    ├── timerft_train_max256s_cot_easyr1.jsonl  # CoT 训练集 (2148)
    ├── tvgbench_val_max256s_easyr1.jsonl       # No CoT 验证集 (800)
    └── tvgbench_val_max256s_cot_easyr1.jsonl   # CoT 验证集 (800)

verl/reward_function/
├── temporal_grounding_reward.py   # 新增: TG 专用 IoU reward
└── mixed_proxy_reward.py          # 已注册 temporal_grounding
```
