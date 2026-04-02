# Seg-AOT (Seg Annotation-based Action Ordering Task) 消融实验

基于 YouCook2 三层分割标注（L2 events / L3 grounding），直接从 seg annotation 文本标签构造视频时序理解代理任务，
消除独立的 VLM captioning 步骤，数据流统一为一套 seg annotation 源。

## 研究问题

**任务粒度（action vs event）和方向（V2T vs T2V）如何影响 RL 训练的时序理解能力？**

2×2 消融矩阵（粒度 × 方向）:

|  | V2T (给视频选文本) | T2V (给文本选视频) |
|--|------------------|------------------|
| **Action level (L3)** | exp1: `action_v2t` | exp2: `action_t2v` |
| **Event level (L2)** | exp3: `event_v2t` | exp4: `event_t2v` |

---

## 四种任务设计

| Exp | problem_type | 粒度 | 方向 | 输入 | 问题 | 选项 |
|-----|---|---|---|---|---|---|
| 1 | `seg_aot_action_v2t` | L3 atomic | V2T | L3 event clip (单视频) | 哪个动作列表正确？ | A/B (forward vs reversed) |
| 2 | `seg_aot_action_t2v` | L3 atomic | T2V | forward 动作列表 + 2 个 L3 clips | 哪个 clip 包含这些动作？ | A/B |
| 3 | `seg_aot_event_v2t` | L2 event | V2T | L2 window clip (单视频) | 哪个事件列表正确？ | A/B/C (forward/shuffle/reversed) |
| 4 | `seg_aot_event_t2v` | L2 event | T2V | forward 事件列表 + 3 个 L2 clips | 哪个 clip 包含这些事件？ | A/B/C |

### 关键对比

| 对比 | 测试什么 |
|------|---------|
| exp1 vs exp2 | V2T vs T2V 在 action 粒度的效果 |
| exp3 vs exp4 | V2T vs T2V 在 event 粒度的效果 |
| exp1 vs exp3 | Action 粒度 vs Event 粒度（V2T 方向） |
| exp2 vs exp4 | Action 粒度 vs Event 粒度（T2V 方向） |

---

## 数据来源

| 层级 | 数据来源 | 视频路径 | 文本标签 |
|------|---------|---------|---------|
| L3 (action) | `level3.grounding_results[].sub_action` | `clips/L3/{key}_L3_ev{id}_{cs}_{ce}.mp4` | 原子动作描述 |
| L2 (event) | `level2.events[].instruction` | `clips/L2/{key}_L2_w{ws}_{we}.mp4` | 事件描述 |

干扰项策略:
- **action_t2v**: 同 `clip_key` 内另一 L2 event 的 L3 clip（同视频不同事件，自然不同顺序）
- **event_t2v**: 全局随机取其他 clip_key 的 L2 window clips（不同视频）

---

## 数据构建

```bash
# 构建单个任务的数据
python proxy_data/temporal_aot/build_aot_from_seg.py \
  --annotation-dir ${ANNOTATION_DIR} \
  --clip-dir-l2 ${CLIP_DIR_L2} \
  --clip-dir-l3 ${CLIP_DIR_L3} \
  --output-dir /path/to/output \
  --tasks action_v2t \
  --min-actions 3 --total-val 200 --train-per-task 500 \
  --complete-only

# 构建所有 4 种任务（event_t2v 自动包含 event_v2t 池）
python proxy_data/temporal_aot/build_aot_from_seg.py \
  --annotation-dir ${ANNOTATION_DIR} \
  --clip-dir-l2 ${CLIP_DIR_L2} --clip-dir-l3 ${CLIP_DIR_L3} \
  --output-dir /path/to/output \
  --tasks action_v2t action_t2v event_v2t event_t2v \
  --complete-only
```

---

## 运行

```bash
# 单个实验（数据首次运行时自动构建）
bash local_scripts/aot_ablations/exp_seg_action_v2t.sh
bash local_scripts/aot_ablations/exp_seg_action_t2v.sh
bash local_scripts/aot_ablations/exp_seg_event_v2t.sh
bash local_scripts/aot_ablations/exp_seg_event_t2v.sh

# 自定义 exp 名称
EXP_NAME=my_exp SEG_TASKS="action_v2t action_t2v" \
  bash local_scripts/aot_ablations/launch_seg_train.sh
```

---

## 文件结构

```
local_scripts/aot_ablations/
├── README.md                        # 本文件
├── common.sh                        # 共用超参数 (含 ANNOTATION_DIR / CLIP_DIR_L2/L3)
├── launch_seg_train.sh              # 统一训练入口 (数据构建 → 训练)
├── exp_seg_action_v2t.sh            # exp1: action V2T binary
├── exp_seg_action_t2v.sh            # exp2: action T2V binary
├── exp_seg_event_v2t.sh             # exp3: event V2T 3-way
└── exp_seg_event_t2v.sh             # exp4: event T2V 3-way

proxy_data/temporal_aot/
├── build_aot_from_seg.py            # [ACTIVE] 从 seg annotation JSON 构建 6 种 MCQ（L1/L2/L3）
├── rebalance_aot_answers.py         # [ACTIVE] 答案重平衡
├── prompts.py                       # [ACTIVE] prompt 模板（旧版管线用）
└── legacy/                          # [Archive] 旧管线脚本（已废弃）

verl/reward_function/
└── mixed_proxy_reward.py            # seg_aot_* → _choice_reward (A/B/C 精确匹配)
```

---

## 共用超参数

| 参数 | 值 |
|------|----|
| Model | Qwen3-VL-4B-Instruct |
| Algorithm | `ema_grpo` |
| LR | 5e-7 (cosine) |
| MAX_STEPS | 60 |
| ROLLOUT_N | 8 |
| Train per task | 500 |
| Val total | 200 |

完整参数见 [common.sh](common.sh)。
