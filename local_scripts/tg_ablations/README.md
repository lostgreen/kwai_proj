# Temporal Grounding 消融实验

基于 Time-R1 的 TimeRFT 训练数据，在 VideoProxy 256 frames / 2fps 框架下进行 temporal grounding RL 消融。

## 研究问题

**Chain-of-Thought 是否有助于时间定位？** 模型先在 `<think>` 中分析视频时间线，是否能比直接输出更精确地定位事件？

## 实验矩阵

| Exp | 名称 | CoT | Response 长度 | 说明 |
|-----|------|-----|---------------|------|
| 1 | `tg_ablation_exp1_no_cot_v2` | ✗ | 256 | 直接输出 `<events>[[s, e]]</events>` |
| 2 | `tg_ablation_exp2_cot_v2` | ✓ | 1024 | 先 `<think>` 分析再输出时间段 |

两组实验仅 CoT 与否和 response 长度不同，其余超参数完全一致（受控消融）。

---

## 数据

- **训练集**: TimeRFT 2.5K → 过滤 >256s → 过滤坏视频 → **~2148 条**
- **验证集**: TVGBench ≤256s → 过滤坏视频 → difficulty-stratified sampling → **200 条**
- **视频路径**: `/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/TimeR1-Dataset/`

### 数据格式

每条 JSONL 样本包含:

```json
{
  "prompt": "Watch the following video...",
  "answer": "<events>\n[1.0, 4.0]\n</events>",
  "videos": ["/path/to/video.mp4"],
  "problem_type": "temporal_grounding",
  "metadata": {
    "duration": 93.29,
    "timestamp": [1.0, 4.0],
    "sentence": "事件描述",
    "source": "cosmo",
    "difficulty": 50.0
  }
}
```

### Prompt 模板差异

- **No CoT**: "Output format (strictly follow this): `<events>[start_time, end_time]</events>`"
- **CoT**: 额外添加 "First, think step by step inside `<think></think>` tags. Describe what happens at different time periods..."

CoT 数据由 `convert_nocot_to_cot.py` 从 no_cot JSONL 自动转换（仅替换 prompt 指令部分）。

---

## Reward: IoU × distance_penalty (iou_v2)

```
reward = IoU(pred, gt) × (1 - |gt_s - pred_s| / duration) × (1 - |gt_e - pred_e| / duration)
```

**为什么不用纯 IoU？** 纯 IoU 存在"懒惰最优解"——模型输出覆盖整个视频的超长片段即可稳定获得 `IoU ≈ gt_length / duration ≈ 0.2`，无需精确定位。distance_penalty 通过惩罚端点偏移打破此局部最优：

| 预测 | 纯 IoU | iou_v2 |
|------|--------|--------|
| 整个视频 `[0, 100]` (GT=`[10, 30]`, dur=100) | 0.200 | **0.054** |
| 稍宽预测 `[0, 10]` (GT=`[1, 4]`, dur=93) | 0.300 | **0.278** |
| 精确预测 `[1, 4]` | 1.000 | **1.000** |

### 反作弊

- 多个 `<events>` 标签 → 0 分
- 出现 `[数字-数字]` 格式 → 0 分（防止猜测连字符格式）

---

## 共用超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Model | Qwen3-VL-4B-Instruct | |
| Algorithm | `ema_grpo` | EMA-normalized GRPO advantage |
| LR | 5e-7 | cosine warmup/decay, warmup_ratio=0.1 |
| KL loss | 启用 | |
| Online filtering | 启用 | filter_low=0.01, filter_high=0.99 |
| Rollout N | 8 | |
| Temperature | 0.7, top_p=0.9 | |
| Video FPS | 2.0, max 256 frames | |
| Entropy coeff | 0.005 | |
| Clip ratio | low=0.2, high=0.3 | 非对称裁剪 |
| Max steps | 60 | |
| Val freq | 10 steps | |
| Save freq | 20 steps | |

超参数与 AOT 消融实验完全对齐，便于跨任务对比。

---

## 运行

```bash
cd /path/to/train

# 实验 1: No CoT
bash local_scripts/tg_ablations/exp1_no_cot.sh

# 实验 2: CoT
bash local_scripts/tg_ablations/exp2_cot.sh
```

超参数可通过环境变量覆盖：

```bash
LR=1e-6 bash local_scripts/tg_ablations/exp1_no_cot.sh
MAX_STEPS=100 bash local_scripts/tg_ablations/exp2_cot.sh
```

---

## Checkpoint 路径

```
/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/temporal_grounding/ablations/
├── tg_ablation_exp1_no_cot_v2/
│   ├── run_YYYYMMDD_HHMMSS.log
│   ├── global_step_20/ ...
│   └── ray_logs/
└── tg_ablation_exp2_cot_v2/
    └── ...
```

---

## 文件结构

```
local_scripts/tg_ablations/
├── README.md               # 本文件
├── common.sh               # 共用超参数（与 aot_ablations 对齐）
├── launch_train.sh          # 统一训练入口
├── exp1_no_cot.sh           # 实验 1: 无 CoT
└── exp2_cot.sh              # 实验 2: 有 CoT

proxy_data/temporal_grounding/
├── DESIGN.md                # 设计文档
├── build_dataset.py          # annotation JSON → EasyR1 JSONL
├── convert_nocot_to_cot.py   # no_cot JSONL → CoT JSONL（替换 prompt）
└── data/
    ├── timerft_train_max256s_easyr1_clean.jsonl          # No CoT 训练集
    ├── timerft_train_max256s_cot_easyr1_clean.jsonl      # CoT 训练集
    ├── tvgbench_val_max256s_easyr1_200_clean.jsonl       # No CoT 验证集 (200)
    └── tvgbench_val_max256s_cot_easyr1_200_clean.jsonl   # CoT 验证集 (200)

verl/reward_function/
├── temporal_grounding_reward.py   # TG 专用 iou_v2 reward
└── mixed_proxy_reward.py          # 多任务 reward 路由（已注册 temporal_grounding）
```
