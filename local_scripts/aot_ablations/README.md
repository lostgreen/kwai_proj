# AoT 消融实验启动指南

## 目录结构

```
/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/          ← AOT_DATA_ROOT
├── aot_event_manifest.jsonl          # Step 1 产出（所有实验共享）
├── caption_pairs.jsonl               # Step 2 产出（所有实验共享）
├── forward_captions.jsonl
├── reverse_captions.jsonl
├── shuffle_captions.jsonl
├── composite_clips/
├── reverse_clips/
├── shuffle_clips/
└── ablations/                        # 消融实验数据目录
    ├── exp1/                         # 实验 1 独立目录
    │   ├── v2t_binary.jsonl          # Step A: MCQ 构造产出
    │   ├── mixed_train.jsonl         # Step B: 混合 temporal_seg
    │   ├── mixed_val.jsonl
    │   ├── mixed_train.offline_filtered.jsonl   # Step C: 离线筛选后
    │   └── offline_filter_report.jsonl
    ├── exp2/
    │   ├── v2t_binary.jsonl
    │   ├── v2t_4way.jsonl
    │   ├── mixed_train.jsonl
    │   ├── mixed_val.jsonl
    │   ├── mixed_train.offline_filtered.jsonl
    │   └── offline_filter_report.jsonl
    ├── exp3/ ... exp6/               # 同上，依实验不同产出文件略有差异
    └── ...
```

> MCQ 构造数据和离线筛选结果都存放在各实验自己的目录下，方便：
> - 每个实验的数据独立管理，不互相干扰
> - 筛选结果可直接复用（第二次跑训练时自动跳过构造和筛选）
> - 服务器上长期保留，随时对比或重新训练

---

## 快速启动

在 repo 根目录下直接执行对应实验脚本：

```bash
# 实验 1：V2T-Binary
bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 实验 2：V2T-4way
bash local_scripts/aot_ablations/exp2_v2t_4way.sh

# 实验 3：T2V-Binary
bash local_scripts/aot_ablations/exp3_t2v_binary.sh

# 实验 4：T2V-4way
bash local_scripts/aot_ablations/exp4_t2v_4way.sh

# 实验 5：Mixed-Binary（V2T + T2V，各 A/B）
bash local_scripts/aot_ablations/exp5_mixed_binary.sh

# 实验 6：Mixed-4way（V2T+T2V 4-way 联合训练）
bash local_scripts/aot_ablations/exp6_mixed_4way.sh

# 实验 7：V2T-4way 仅 forward_video，纯 distractor 难度增强
bash local_scripts/aot_ablations/exp7_v2t_4way_fwdonly.sh
```

每个脚本内部依次执行：

| 步骤 | 内容 | 幂等 |
|------|------|------|
| **Step A** | `build_aot_mcq.py` 构造 MCQ | ✅ 若 `mixed_train.jsonl` 已存在则跳过 |
| **Step B** | `mix_aot_with_youcook2.py` 混合 temporal_seg | ✅ 同上 |
| **Step C** | `offline_rollout_filter.py` 离线过滤 | ✅ 若 filtered 文件已存在则跳过 |
| **Step D** | 自动推断 task weights | — |
| **Step E** | `verl.trainer.main` 启动 RL 训练 | — |

---

## 强制重建数据

```bash
# 强制重做 MCQ 构造 + mix（即使 mixed_train.jsonl 已存在）
FORCE_BUILD=true bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 强制重做离线筛选（即使 filtered 文件已存在）
FORCE_FILTER=true bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 两者都强制重做
FORCE_BUILD=true FORCE_FILTER=true bash local_scripts/aot_ablations/exp1_v2t_binary.sh
```

使用场景：
- 标注数据更新后（`caption_pairs.jsonl` 有变化）→ 加 `FORCE_BUILD=true`
- 想调整过滤严格程度 → 加 `FORCE_FILTER=true`

---

## 覆盖超参数

所有超参数都可以通过环境变量在启动时覆盖，无需修改脚本文件：

```bash
# 修改数据根目录（默认 /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot）
AOT_DATA_ROOT=/your/path bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 修改模型路径
MODEL_PATH=/your/model/path bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 修改 MCQ 样本数（当前默认 1000）
MCQ_MAX_SAMPLES=2000 bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 修改学习率
LR=1e-6 bash local_scripts/aot_ablations/exp1_v2t_binary.sh

# 修改 GPU 配置（当前默认 8 GPU, TP=2）
N_GPUS_PER_NODE=4 TP_SIZE=2 bash local_scripts/aot_ablations/exp1_v2t_binary.sh
```

所有可覆盖变量见 [common.sh](common.sh)。

---

## 7 组实验说明

> 详细的任务构造（每个问题是什么、选项来源、正确答案类型）见 [docs/aot_task_construction.md](../../docs/aot_task_construction.md)

| # | 脚本名 | AoT 训练数据 | 选项数 | 实验目的 |
|---|--------|-------------|--------|---------|
| 1 | `exp1_v2t_binary.sh`  | `aot_v2t` | A/B | 基线：给视频选 caption（forward vs reverse 2选1） |
| 2 | `exp2_v2t_4way.sh`   | `aot_4way_v2t`（含 forward+shuffle 视频） | A/B/C/D | 4选1，正确答案为 forward 或 shuffle caption |
| 3 | `exp3_t2v_binary.sh` | `aot_t2v` | A/B | 基线：给 caption 选视频片段（2选1） |
| 4 | `exp4_t2v_4way.sh`   | `aot_4way_t2v` | A/B/C/D | 4选1，从 4 段视频中找匹配 caption 的那段 |
| 5 | `exp5_mixed_binary.sh` | `aot_v2t` + `aot_t2v` | A/B | V2T+T2V binary 联合训练 |
| 6 | `exp6_mixed_4way.sh` | `aot_4way_v2t` + `aot_4way_t2v` | A/B/C/D | V2T+T2V 4-way 联合训练 |
| **7** | `exp7_v2t_4way_fwdonly.sh` | `aot_4way_v2t`（**仅 forward_video**） | A/B/C/D | 4选1，正确答案始终为 forward caption，无 shuffle video |

关键对比：
- **exp1 vs exp7**：保持"只看正放视频"，选项难度从 2 增到 4（加了 shuffle/hard_neg distractor），纯 distractor 增强效果
- **exp7 vs exp2**：都是 4-way，但 exp2 额外包含 shuffle_video 样本，比较"加入杂乱时序识别训练"的额外收益
- **exp1 vs exp2**：binary 到完整 4-way 的整体提升（包含两个变化：distractor 数 + shuffle video）
- **exp3 vs exp4**：T2V 方向的相同对比
- **exp1 vs exp3**：V2T vs T2V 哪个方向更有效
- **exp5 vs exp6**：binary 混合 vs 4-way 混合的增益

---

## 数据文件说明

筛选完成后，各实验目录下的关键文件：

| 文件 | 说明 |
|------|------|
| `v2t_binary.jsonl` | binary V2T 样本（exp1/exp5 使用） |
| `t2v_binary.jsonl` | binary T2V 样本（exp3/exp5 使用） |
| `v2t_4way.jsonl` | 4-way V2T 原始样本（exp2/exp6/exp7 使用） |
| `v2t_4way_fwd_only.jsonl` | 仅 forward_video 的 4-way V2T（exp7 由 launch_train.sh 过滤生成） |
| `t2v_4way.jsonl` | 4-way T2V 原始样本（exp4/exp6 使用） |
| `mixed_train.jsonl` | 完整训练集（Step B 输出） |
| `mixed_val.jsonl` | 验证集 |
| `mixed_train.offline_filtered.jsonl` | 离线筛选后（Step C 输出） |
| `mixed_train.offline_filtered.balanced.jsonl` | **答案重平衡后（Step C+ 输出）← 训练实际用这个** |
| `offline_filter_report.jsonl` | 筛选被丢弃的样本记录 |

---

## 并行运行多个实验

在不同的 tmux 窗口或机器上同时跑多组：

```bash
# 机器 A：跑实验 1 和 2
tmux new-session -d -s exp1 "bash local_scripts/aot_ablations/exp1_v2t_binary.sh 2>&1 | tee /tmp/exp1.log"
tmux new-session -d -s exp2 "bash local_scripts/aot_ablations/exp2_v2t_4way.sh   2>&1 | tee /tmp/exp2.log"

# 机器 B：跑实验 3 和 4
tmux new-session -d -s exp3 "bash local_scripts/aot_ablations/exp3_t2v_binary.sh 2>&1 | tee /tmp/exp3.log"
tmux new-session -d -s exp4 "bash local_scripts/aot_ablations/exp4_t2v_4way.sh   2>&1 | tee /tmp/exp4.log"
```

> 注意：同一台机器上同时跑多个实验时需要有足够的 GPU 资源。Step A-C（MCQ 构造和离线筛选）会占用全部 GPU；训练阶段也会占满。建议分时或分机跑。

---

## 文件结构一览

```
local_scripts/aot_ablations/
├── README.md              ← 本文件
├── common.sh              ← 所有实验共用超参数（学习率、batch size 等）
├── launch_train.sh        ← 完整 pipeline（Step A-E），被各 exp 脚本 source
├── exp1_v2t_binary.sh
├── exp2_v2t_4way.sh
├── exp3_t2v_binary.sh
├── exp4_t2v_4way.sh
├── exp5_mixed_binary.sh
└── exp6_mixed_4way.sh
```
