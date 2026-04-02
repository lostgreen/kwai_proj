# Seg-AOT (Seg Annotation-based Action Ordering Task) 消融实验

基于三层分割标注（L1 phases / L2 events / L3 actions），从 seg annotation 构造时序理解代理任务。
原子 clips 拼接为 fwd/shuf/rev 视频，消除时长差异和未标注间隙导致的 shortcut。

## 研究问题

**V2T（给视频选文本）和 T2V（给文本选视频）哪种方向更有效？**

2 组实验，每组包含全部 3 层粒度，L1:L2:L3 = 1:2:2 采样比例（侧重 L2/L3）：

| 实验 | 方向 | 任务 | 总训练量 |
|------|------|------|---------|
| `seg_aot_v2t` | V2T | `phase_v2t` + `event_v2t` + `action_v2t` | 1000 (200+400+400) |
| `seg_aot_t2v` | T2V | `phase_t2v` + `event_t2v` + `action_t2v` | 1000 (200+400+400) |

---

## 任务设计（全部 3-way MCQ: A/B/C）

| problem_type | 粒度 | 方向 | 视频输入 | 文本 | 选项 |
|---|---|---|---|---|---|
| `seg_aot_phase_v2t` | L1 phase | V2T | 1 个 fwd-concat 视频 | 3 种文本顺序 (fwd/shuf/rev) | A/B/C |
| `seg_aot_phase_t2v` | L1 phase | T2V | 3 个拼接视频 (fwd/shuf/rev) | round-robin 文本 | A/B/C |
| `seg_aot_event_v2t` | L2 event | V2T | 1 个 fwd-concat 视频 | 3 种文本顺序 | A/B/C |
| `seg_aot_event_t2v` | L2 event | T2V | 3 个拼接视频 | round-robin 文本 | A/B/C |
| `seg_aot_action_v2t` | L3 action | V2T | 1 个 fwd-concat 视频 | 3 种文本顺序 | A/B/C |
| `seg_aot_action_t2v` | L3 action | T2V | 3 个拼接视频 | round-robin 文本 | A/B/C |

T2V 每组产 1 条 record，文本顺序 round-robin (fwd→shuf→rev) 确保均衡。

---

## 流程

### Step 0: 切分原子 clips

```bash
bash local_scripts/aot_ablations/prepare_clips.sh
```

### Step 1: 训练（数据首次运行时自动构建）

```bash
# V2T 实验
bash local_scripts/aot_ablations/exp_v2t.sh

# T2V 实验
bash local_scripts/aot_ablations/exp_t2v.sh

# 自定义
EXP_NAME=my_exp SEG_TASKS="phase_v2t event_v2t action_v2t" TRAIN_TOTAL=500 \
  bash local_scripts/aot_ablations/launch_seg_train.sh
```

---

## 文件结构

```
local_scripts/aot_ablations/
├── README.md
├── common.sh                # 共用超参数
├── prepare_clips.sh         # 独立视频切分
├── launch_seg_train.sh      # 统一训练入口 (concat + build + train)
├── exp_v2t.sh               # V2T 实验 (L1+L2+L3)
└── exp_t2v.sh               # T2V 实验 (L1+L2+L3)

proxy_data/youcook2_seg/
├── prepare_all_clips.py     # 切分原子 clips (三条流水线通用)
└── temporal_aot/
    └── build_aot_from_seg.py  # 构建 MCQ 数据 (concat + JSONL)
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
| Train total | 1000 |
| Level ratio | 1:2:2 (L1:L2:L3) |
| Val total | 200 |

完整参数见 [common.sh](common.sh)。
