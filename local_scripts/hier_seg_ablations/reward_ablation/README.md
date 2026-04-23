# Reward Ablation — Hier Seg Multi-Task RL

当前 reward 消融默认跑三组实验:

- `R1`: Hungarian `F1-IoU` + `GRPO`
- `R4`: `Segment Matching` + `GRPO`
- `R1_EMA`: Hungarian `F1-IoU` + `EMA-GRPO`

三组实验都固定为:

- 模型: `Qwen3-VL-4B-Instruct`
- 主线算法: `GRPO`
- 保险对照: `R1_EMA` 使用 `EMA-GRPO`
- `LR=1e-6`
- `KL_COEF=0.001`
- `ENTROPY_COEFF=0.0`
- `MAX_FRAMES=256`
- `MAX_PIXELS=65536`
- `ONLINE_FILTERING=true`
- `TASKS="tg mcq hier_seg"`
- `HIER_TARGET=0`，即 full hier-seg train

## Reward 路由

不要再直接依赖 `mixed_proxy_reward.py` 的默认 hier dispatch。

现在 `R1/R4/R1_EMA` 都统一走:

`verl/reward_function/mixed_proxy_reward_ablation.py:compute_score`

并通过环境变量切换 hier reward:

- `R1`: `HIER_REWARD_MODE=f1_iou`
- `R4`: `HIER_REWARD_MODE=seg_match`
- `R1_EMA`: `HIER_REWARD_MODE=f1_iou`

这样 MCQ / TG 仍然走原来的 mixed-task reward，只有 hier-seg reward 被替换。

## 数据路径

脚本会优先使用共享 frame-list manifest:

- `/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/train/train_all_shared_frames.jsonl`
- `/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/train/val_all_shared_frames.jsonl`

如果共享 manifest 不存在，则自动回退到 `multi_task_common.sh` 里的原始 `train_all.jsonl / val_all.jsonl`。

## 用法

```bash
# 依次跑 R1 + R4 + R1_EMA
bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh

# 只跑 R1
EXPS="R1" bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh

# 只跑 R4
EXPS="R4" bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh

# 只跑 R1_EMA
EXPS="R1_EMA" bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh

# 调试
MAX_STEPS=10 EXPS="R1" bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh
```

## 常用覆盖项

```bash
# 显式指定 shared-frame train manifest
HIER_TRAIN=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/train/train_all_shared_frames.jsonl \
EXPS="R1" \
bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh

# 自定义学习率 / max pixels
LR=2e-6 MAX_PIXELS=98304 EXPS="R4" \
bash local_scripts/hier_seg_ablations/reward_ablation/run_reward_ablation.sh
```

## 说明

- `R3` 已从当前 reward ablation 入口移除，避免和 `R1` 混淆。
- 现在默认 `R1/R4` 先用纯 `GRPO`，方便先看 reward 本身会不会带来训练问题。
- `R1_EMA` 只作为保险对照，不和 `R4` 组成主消融对。
- 如果你之前跑过老版本 `reward_ablation_R1_f1iou` 之类的实验目录，这次默认 `EXP_NAME` 已更新，避免复用到旧的 `train.jsonl` 采样缓存。
