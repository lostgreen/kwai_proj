# AoT (Action Ordering Task) 消融实验

基于 YouCook2 烹饪事件片段，构造视频时序理解代理任务（正放/倒放/乱序），通过 RL 训练提升模型的时序理解能力。

## 研究问题

**什么任务形式和选项粒度最能通过 RL 训练激活下游时序理解能力？**

消融系统地变化两个正交维度：

|  | Binary (2选1) | 3-way (3选1) | Binary + 3-way 混合 |
|--|--------------|-------------|-------------------|
| **V2T only** | exp1 | exp2 | exp7 |
| **T2V only** | exp3 | exp4 | exp9 |
| **V2T + T2V** | exp5 | exp6 | exp8 |

---

## 数据来源与构造

**数据源**: YouCook2 烹饪事件片段 → 正放(forward)/倒放(reverse)/乱序(shuffle) 三种变体

**Pipeline** (5 步，在 `proxy_data/temporal_aot/` 目录):

| 步骤 | 脚本 | 产出 |
|------|------|------|
| Step 1 | `build_event_aot_data.py` | manifest + reverse/shuffle 视频 |
| Step 2 | `annotate_event_captions.py` | forward/reverse/shuffle caption (Gemini API) |
| Step 2.5 | `check_and_refine_captions.py` | 过滤文本不可区分的 caption 对 |
| Step 3 | `build_aot_mcq.py` | 4 种 MCQ 训练样本 |
| Step 4 | `rebalance_aot_answers.py` | 答案分布平衡（防位置偏差） |

### 4 种 MCQ 类型

| problem_type | 方向 | 选项 | 描述 |
|---|---|---|---|
| `aot_v2t` | V2T | A/B | 给视频选 caption（forward vs reverse） |
| `aot_t2v` | T2V | A/B | 给 caption 选视频段 |
| `aot_3way_v2t` | V2T | A/B/C | 给视频选 caption（含 shuffle distractor） |
| `aot_3way_t2v` | T2V | A/B/C | 给 caption 选视频（含 shuffle distractor） |

### 反作弊措施

- 选项顺序随机化（A/B/C 插槽随机分配）
- `direction_clear` 过滤循环动作（搅拌、揉面）
- Caption 置信度阈值 ≥ 0.6
- 正/反/乱序 caption 长度保持可比（防长度捷径）

---

## Reward

所有 AOT 任务使用**精确匹配**（binary 0/1）:

| 情况 | overall | format | accuracy |
|------|---------|--------|----------|
| 答案正确且格式正确 | 1.0 | 1.0 | 1.0 |
| 答案错误但格式正确 | 0.0 | 1.0 | 0.0 |
| 格式不正确 | 0.0 | 0.0 | 0.0 |

格式要求：回复必须包含 `<answer>X</answer>` 标签（X 为单个大写字母）。

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
| Max response len | 1024 | |
| Entropy coeff | 0.005 | |
| Clip ratio | low=0.2, high=0.3 | 非对称裁剪 |
| Max steps | 60 | |
| Val freq | 10 steps | |
| Save freq | 20 steps | |

超参数与 TG 消融实验完全对齐，便于跨任务对比。

---

## 目录结构

```
/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/          <- AOT_DATA_ROOT
├── aot_event_manifest.jsonl          # Step 1 产出（所有实验共享）
├── refined_caption_pairs.jsonl       # Step 2 产出（所有实验共享）
├── forward_captions.jsonl
├── reverse_captions.jsonl
├── shuffle_captions.jsonl
├── composite_clips/
├── reverse_clips/
├── shuffle_clips/
└── ablations_refined/                # 消融实验数据目录
    ├── exp1/ ... exp9/               # 各实验独立目录
    │   ├── v2t_binary.jsonl          # Step A: MCQ 构造产出
    │   ├── mixed_train.jsonl         # Step B: 训练集
    │   ├── mixed_val.jsonl           # 验证集
    │   ├── mixed_train.offline_filtered.jsonl   # Step C: 离线筛选后
    │   └── offline_filter_report.jsonl
    └── ...
```

---

## 快速启动

在 repo 根目录下直接执行对应实验脚本：

```bash
# 单方向 x 单类型（基线）
bash local_scripts/aot_ablations/exp1_v2t_binary.sh
bash local_scripts/aot_ablations/exp2_v2t_4way.sh
bash local_scripts/aot_ablations/exp3_t2v_binary.sh
bash local_scripts/aot_ablations/exp4_t2v_4way.sh

# 双方向 x 单类型
bash local_scripts/aot_ablations/exp5_mixed_binary.sh
bash local_scripts/aot_ablations/exp6_mixed_4way.sh

# 单方向 x 混合类型
bash local_scripts/aot_ablations/exp7_v2t_binary_3way_mixed.sh
bash local_scripts/aot_ablations/exp9_t2v_binary_3way_mixed.sh

# 全混合
bash local_scripts/aot_ablations/exp8_all_mixed.sh
```

每个脚本（exp1-6）内部依次执行：

| 步骤 | 内容 | 幂等 |
|------|------|------|
| **Step A** | `build_aot_mcq.py` 构造 MCQ | 若 `mixed_train.jsonl` 已存在则跳过 |
| **Step B** | 合并/切分训练集和验证集 | 同上 |
| **Step C** | `offline_rollout_filter.py` 离线过滤 | 若 filtered 文件已存在则跳过 |
| **Step D** | 自动推断 task weights | — |
| **Step E** | `verl.trainer.main` 启动 RL 训练（含在线过滤） | — |

exp7-9 使用 `launch_train_cross_exp.sh`，从其他实验的已过滤数据中采样组合，跳过 Step A-C。

---

## 9 组实验说明

### 单方向 x 单类型（4 组基线）

| # | 脚本名 | problem_type | 选项数 | 实验目的 |
|---|--------|-------------|--------|---------|
| 1 | `exp1_v2t_binary.sh` | `aot_v2t` | A/B | 给视频选 caption（forward vs reverse 2选1） |
| 2 | `exp2_v2t_4way.sh` | `aot_3way_v2t` | A/B/C | 给视频选 caption（3选1，含 shuffle distractor） |
| 3 | `exp3_t2v_binary.sh` | `aot_t2v` | A/B | 给 caption 选视频（forward vs reverse 2选1） |
| 4 | `exp4_t2v_4way.sh` | `aot_3way_t2v` | A/B/C | 给 caption 选视频（3选1，含 shuffle distractor） |

### 双方向 x 单类型（2 组）

| # | 脚本名 | problem_type | 选项数 | 实验目的 |
|---|--------|-------------|--------|---------|
| 5 | `exp5_mixed_binary.sh` | `aot_v2t` + `aot_t2v` | A/B | V2T+T2V binary 联合训练 |
| 6 | `exp6_mixed_4way.sh` | `aot_3way_v2t` + `aot_3way_t2v` | A/B/C | V2T+T2V 3-way 联合训练 |

### 单方向 x 混合类型（2 组）

| # | 脚本名 | problem_type | 选项数 | 实验目的 |
|---|--------|-------------|--------|---------|
| 7 | `exp7_v2t_binary_3way_mixed.sh` | `aot_v2t` + `aot_3way_v2t` | 混合 | V2T binary+3way 混合，测试互补性 |
| 9 | `exp9_t2v_binary_3way_mixed.sh` | `aot_t2v` + `aot_3way_t2v` | 混合 | T2V binary+3way 混合，测试互补性 |

### 全混合（1 组）

| # | 脚本名 | problem_type | 选项数 | 实验目的 |
|---|--------|-------------|--------|---------|
| 8 | `exp8_all_mixed.sh` | 全部 4 种 | 混合 | 最大任务多样性（4种类型等量混合） |

### 关键对比

| 对比 | 测试什么 |
|------|---------|
| exp1 vs exp2 | V2T 方向: binary(2选1) vs 3-way(3选1) |
| exp3 vs exp4 | T2V 方向: binary vs 3-way |
| exp1 vs exp3 | V2T vs T2V 哪个方向更有效 |
| exp2 vs exp4 | 3-way 下 V2T vs T2V |
| exp5 vs exp6 | binary 混合 vs 3-way 混合 |
| exp7 vs exp9 | V2T 同方向混合 vs T2V 同方向混合 |
| exp7 vs exp1/exp2 | binary+3way 混合 vs 单类型 |
| exp8 vs exp5/exp6/exp7/exp9 | 全混合 vs 局部混合 |

---

## 强制重建数据

```bash
# 强制重做 MCQ 构造 + mix
FORCE_BUILD=true bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 强制重做离线筛选
FORCE_FILTER=true bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 两者都强制重做
FORCE_BUILD=true FORCE_FILTER=true bash local_scripts/aot_ablations/exp1_v2t_binary.sh
```

---

## 覆盖超参数

所有超参数都可以通过环境变量在启动时覆盖，无需修改脚本文件：

```bash
# 修改学习率（当前默认 5e-7）
LR=1e-6 bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 修改模型路径
MODEL_PATH=/your/model/path bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 修改 GPU 配置（当前默认 8 GPU, TP=2）
N_GPUS_PER_NODE=4 TP_SIZE=2 bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 关闭在线过滤 / 恢复 curate
ONLINE_FILTERING=false SKIP_CURATE=false bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 算法/batching 消融（多任务实验可用）
ADV_ESTIMATOR=grpo bash local_scripts/aot_ablations/exp5_mixed_binary.sh
TASK_HOMOGENEOUS=false bash local_scripts/aot_ablations/exp5_mixed_binary.sh
```

所有可覆盖变量见 [common.sh](common.sh)。

---

## 数据文件说明

各实验目录下的关键文件：

| 文件 | 说明 |
|------|------|
| `v2t_binary.jsonl` / `t2v_binary.jsonl` | binary MCQ 样本 |
| `v2t_3way.jsonl` / `t2v_3way.jsonl` | 3-way MCQ 样本 |
| `mixed_train.jsonl` | 完整训练集 |
| `mixed_val.jsonl` | 验证集（5% 切分） |
| `mixed_train.offline_filtered.jsonl` | 离线筛选后（开启在线过滤时直接用这个训练） |
| `offline_filter_report.jsonl` | 筛选报告（含 mean_reward 等指标） |
| `mixed_train.curated_1000.balanced.jsonl` | curate+rebalance 后（关闭在线过滤时用这个训练） |

---

## 并行运行多个实验

```bash
# 机器 A: exp1 -> exp2 -> exp7 (exp7 依赖 exp1+exp2)
tmux new-session -d -s exp1 "bash local_scripts/aot_ablations/exp1_v2t_binary.sh 2>&1 | tee /tmp/exp1.log"

# 机器 B: exp3 -> exp4 -> exp9 (exp9 依赖 exp3+exp4)
tmux new-session -d -s exp3 "bash local_scripts/aot_ablations/exp3_t2v_binary.sh 2>&1 | tee /tmp/exp3.log"

# 机器 C: exp5 -> exp6 -> exp8 (exp8 依赖 exp1-4)
tmux new-session -d -s exp5 "bash local_scripts/aot_ablations/exp5_mixed_binary.sh 2>&1 | tee /tmp/exp5.log"
```

> 同一台机器上同时跑多个实验需要足够的 GPU 资源。建议分时或分机跑。

---

## 文件结构一览

```
local_scripts/aot_ablations/
├── README.md                          <- 本文件
├── common.sh                          <- 共用超参数
├── launch_train.sh                    <- 完整 pipeline（exp1-6 使用）
├── launch_train_cross_exp.sh          <- 跨实验混合 pipeline（exp7-9 使用）
├── curate_1k_samples.py               <- 难度优先采样脚本
├── run_batch.sh                       <- 批量编排（3 台机器并行）
├── exp1_v2t_binary.sh                 # 单方向 x 单类型
├── exp2_v2t_4way.sh
├── exp3_t2v_binary.sh
├── exp4_t2v_4way.sh
├── exp5_mixed_binary.sh               # 双方向 x 单类型
├── exp6_mixed_4way.sh
├── exp7_v2t_binary_3way_mixed.sh      # 单方向 x 混合类型
├── exp9_t2v_binary_3way_mixed.sh
└── exp8_all_mixed.sh                  # 全混合
```
