# Prompt Ablation V2 — 通用层次分割实验

> 目标：探究 prompt 设计（粒度约束 × 推理链 × 领域无关化）对三层时序分割模型的影响。
>
> 变更简介：V2 版本将所有 prompt **领域通用化**（去除 cooking 词汇），并将 **L3 从 query-grounding 改为自由分割**。

---

## 目录结构

```
prompt_ablation/
  prompt_variants_v2.py         提示词模板定义（L1/L2/L3 × V1-V4，共 12 个）
  prompt_design.md              设计文档（消融设计逻辑、各版本对比、待讨论问题）
  prepare_v2_ablation_data.py   数据准备脚本（从源数据生成 train/val JSONL）
  exp_v2_ablation.sh            单次实验启动脚本
  run_v2_ablation.sh            批量运行 V1-V4 脚本
  README.md                     本文档
```

---

## 关键路径

### 源数据路径（只读）

> 在 `common.sh` 中由 `HIER_DATA_ROOT` 控制，默认：
> `<repo_root>/proxy_data/youcook2_seg_annotation/datasets/`

| 文件 | 条数 | 说明 |
|------|------|------|
| `youcook2_hier_L1_train_clipped.jsonl` | 500 | 宏观阶段分割（帧编号） |
| `youcook2_hier_L2_train_clipped.jsonl` | 1898 | 目标事件检测（128s 滑窗，秒级） |
| `youcook2_hier_L3_train_clipped.jsonl` | 3568 | 原子操作（含 sequential + shuffled，准备时只用 sequential） |

### 预处理输出路径（可写）

> 由 `ABLATION_DATA_ROOT` 控制，默认：
> `/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_seg/ablation_data/`

命名规则：`${ABLATION_DATA_ROOT}/hier_seg_v2_<VARIANT>_<LEVELS>/`

示例：
```
hier_seg_v2_V1_L1_L2_L3/train.jsonl
hier_seg_v2_V1_L1_L2_L3/val.jsonl
hier_seg_v2_V4_L2/train.jsonl
hier_seg_v2_V4_L2/val.jsonl
```

### Checkpoint 路径

> 由 `CHECKPOINT_ROOT` 控制，默认：
> `/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/hier_seg/ablations/`

---

## Prompt 变体说明（2×2 设计）

```
                   No-CoT              Structured CoT（3-step think）
                ┌─────────────────┬──────────────────────────────────┐
Minimal         │  V1  (Baseline) │  V3  CoT-only                    │
                │  MAX_RESP=512   │  MAX_RESP=1024                   │
Granularity-    │  V2  Gran-only  │  V4  Gran + CoT (Full)           │
Enhanced        │  MAX_RESP=512   │  MAX_RESP=1024                   │
                └─────────────────┴──────────────────────────────────┘
```

| Variant | 描述 | 相比 V1 新增内容 |
|---------|------|-----------------|
| **V1** | Minimal baseline，全领域通用 | — |
| **V2** | 粒度增强：明确时长先验、段数期望、DO NOT 规则 | +粒度约束 |
| **V3** | 结构化 CoT：3-step think（观察→分组→跳过） | +推理链 |
| **V4** | 粒度 + CoT 完整版 | +粒度约束 +推理链 |

---

## L3 任务变更

| | 旧版（grounding） | 新版 V2（segmentation） |
|---|---|---|
| 输入 | 视频 + action query 列表 | 视频（无文字提示） |
| 输出 | 每条 query 对应的时间段 | 所有原子操作的时间段 |
| reward | position-aligned mean tIoU | **F1-IoU（与 L1/L2 对齐）** |
| problem_type | `temporal_seg_hier_L3` | `temporal_seg_hier_L3_seg` |

---

## 实验运行流程

### 1. 单次实验（推荐先验证一个变体）

```bash
# 仅 L2，V1 baseline
VARIANT=V1 LEVELS="L2" bash local_scripts/hier_seg_ablations/prompt_ablation/exp_v2_ablation.sh

# 三层 + V4（完整版）
VARIANT=V4 LEVELS="L1 L2 L3" bash local_scripts/hier_seg_ablations/prompt_ablation/exp_v2_ablation.sh
```

脚本会自动：
1. 检查 `${ABLATION_DATA_ROOT}/hier_seg_v2_<VARIANT>_<LEVELS>/train.jsonl` 是否存在
2. 若不存在，调用 `prepare_v2_ablation_data.py` 准备数据
3. 调用 `launch_train.sh` 启动 verl 训练

### 2. 批量运行 V1-V4（消融实验全集）

```bash
# 全三层，V1→V2→V3→V4 顺序运行
LEVELS="L1 L2 L3" bash local_scripts/hier_seg_ablations/prompt_ablation/run_v2_ablation.sh

# 仅 L2 对比（更快）
MAX_STEPS=30 LEVELS="L2" bash local_scripts/hier_seg_ablations/prompt_ablation/run_v2_ablation.sh

# 只跑 V1 和 V3（CoT 效果对比）
VARIANTS="V1 V3" LEVELS="L2" bash local_scripts/hier_seg_ablations/prompt_ablation/run_v2_ablation.sh
```

### 3. 只准备数据（不训练）

```bash
# V4 + 全三层
python3 local_scripts/hier_seg_ablations/prompt_ablation/prepare_v2_ablation_data.py \
  --levels L1 L2 L3 \
  --variant V4 \
  --total-val 200 \
  --output-dir /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_seg/ablation_data/hier_seg_v2_V4_L1_L2_L3

# 一次准备所有变体
python3 local_scripts/hier_seg_ablations/prompt_ablation/prepare_v2_ablation_data.py \
  --levels L1 L2 L3 \
  --variant all \
  --total-val 200 \
  --output-dir /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_seg/ablation_data/hier_seg_v2_all
```

### 4. 覆盖默认路径（本地调试用）

```bash
HIER_DATA_ROOT=/your/local/data/path \
ABLATION_DATA_ROOT=/your/output/path \
CHECKPOINT_ROOT=/your/ckpt/path \
MAX_STEPS=5 \
VARIANT=V1 LEVELS="L2" \
bash local_scripts/hier_seg_ablations/prompt_ablation/exp_v2_ablation.sh
```

---

## 消融实验矩阵（推荐对比方式）

```
1. 粒度约束的效果:        V1 vs V2  （固定 No-CoT，看粒度约束的增益）
2. 结构化 CoT 的效果:    V1 vs V3  （固定 Minimal，看推理链的增益）
3. 两者叠加效果:          V4 vs V1  （Full vs Baseline）
4. 交叉验证（加法性）:   V4 vs (V2 + V3) 的和是否成立

5. 领域泛化（跨数据集）: 用 cooking 训练的 V1-V4 在其他程序性视频数据上评估
                          期望 V1 泛化最好，V4 在 cooking 内最优但迁移可能更难
```

---

## 常见问题

**Q: 为什么 L3 的 `problem_type` 改为 `temporal_seg_hier_L3_seg`？**
A: 旧的 `temporal_seg_hier_L3` 对应 position-aligned tIoU（query 顺序对齐），
V2 改为自由分割后需要 F1-IoU（与 L1/L2 对齐），使用新 `_seg` 后缀区分。
两个 problem_type 在 `youcook2_hier_seg_reward.py` 中都路由到同一个 `_l1_l2_reward`。

**Q: 如果想保留旧的 grounding 实验对比怎么办？**
A: 旧数据（`youcook2_hier_L3_train_clipped.jsonl`）和旧 L3 reward 逻辑（`_l3_reward`、
`compute_aligned_iou`）均保留在代码中，但不再作为默认 dispatch。
可以通过手动指定 `problem_type=temporal_seg_hier_L3` 恢复旧行为（reward 代码已更新，需注意）。

**Q: V3/V4 为什么 MAX_RESPONSE_LEN 设为 1024？**
A: `<think>` 块包含 3 步结构化推理（Observations / Grouping / Skip），
预估 ~300-500 extra tokens，512 可能截断推理内容导致训练信号噪声。
