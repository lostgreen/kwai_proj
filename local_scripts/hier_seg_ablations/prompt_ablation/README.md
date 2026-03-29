# Prompt Ablation — 层次分割 Prompt 消融实验

> 目标：量化 prompt 设计对二层时序分割（L2+L3）的影响。
>
> 对比：**原始标注 prompt**（cooking 语义描述） vs **V3 边界判据 prompt**（domain-agnostic, sparse-aware）。
>
> 两组实验均使用 F1-IoU reward（`youcook2_hier_seg_reward.py`），仅改变 prompt。

---

## 目录结构

```
prompt_ablation/
  exp_pa1_original.sh           PA1: 原始标注 prompt 实验
  exp_pa2_v3boundary.sh         PA2: V3 边界判据 prompt 实验
  run_prompt_ablation.sh        批量运行 PA1/PA2
  prompt_variants_v3.py         V3 prompt 模板定义（L2/L3 × V1-V4，共 8 个）
  prepare_prompt_data.py             数据准备脚本（替换 prompt + 生成 train/val）
  prompt_design.md              设计文档（消融逻辑、术语映射）
  README.md                     本文档
```

---

## 实验设计

### 两组对比

| 实验 | Prompt 来源 | 特点 | CoT | MAX_RESP |
|------|------------|------|-----|----------|
| **PA1** | `prompts.py`（原始标注） | 含 cooking 领域词、语义描述式、无稀疏约束 | 无 | 512 |
| **PA2** | `prompt_variants_v3.py` V2 变体 | 边界判据导向、稀疏采样感知、硬规则、领域无关 | 无 | 512 |

### 控制变量

| 维度 | 两组共用 |
|------|---------|
| 模型 | Qwen3-VL-4B-Instruct |
| 数据层 | L2（事件检测）+ L3（自由分割）— **跳过 L1**（warped 标注问题） |
| Reward | F1-IoU（Hungarian 匹配）— `youcook2_hier_seg_reward.py` |
| 算法 | EMA-GRPO, LR=5e-7, cosine decay |
| 训练 | 60 steps, rollout_bs=16, global_bs=16, 8 GPU |

### PA1 vs PA2 Prompt 对比

**L2 (事件检测)**:

| | PA1（原始） | PA2（V3 V2 变体） |
|---|---|---|
| 定义 | "Detect all complete cooking events" | "LOCAL TASK UNIT — boundary = goal completion or object shift" |
| 时长先验 | 无 | "min 5s, expected 2-8 segments" |
| 稀疏约束 | 无 | "sampled at 1-2 fps, do NOT rely on single-frame micro-motions" |
| 合并规则 | 无 | "merge same-tool/goal spans < 5s" |

**L3 (自由分割)**:

| | PA1（原始） | PA2（V3 V2 变体） |
|---|---|---|
| 定义 | "Detect all atomic cooking actions" | "VISIBLE STATE-CHANGE UNIT — boundary = object state or tool contact change" |
| 时长先验 | 无 | "min 2s, max 15s, expected 3-8 segments" |
| 稀疏约束 | 无 | 同上 |
| 切分规则 | 无 | "split if state-change pause > 3s" |

---

## 数据流水线

```
                    ┌────────────────────────────────────┐
                    │  annotation JSONs (只读)           │
                    │  ${ANNOTATION_DIR}/                │
                    │    *.json (L2/L3 标注)             │
                    └────────┬───────────────────────────┘
                             │
                    build_hier_data.py
                    (读取标注 → 生成带原始 prompt 的 JSONL)
                             │
                    ┌────────▼───────────────────────────┐
                    │  base data (中间产物，两组共用)     │
                    │  ${ABLATION_DATA_ROOT}/             │
                    │    hier_seg_base_L2_L3/             │
                    │      train.jsonl  (~800 条)         │
                    │      val.jsonl    (~200 条)         │
                    └────────┬───────────┬───────────────┘
                             │           │
                    PA1 直接采样    PA2 替换 prompt
                    (保留原始)     (prepare_prompt_data.py)
                             │           │
                    ┌────────▼──┐  ┌─────▼──────────────┐
                    │ PA1 data  │  │ PA2 data            │
                    │ train.jsonl│  │ train.jsonl         │
                    │ val.jsonl │  │ val.jsonl           │
                    └───────────┘  └─────────────────────┘
                             │           │
                     launch_train.sh (verl.trainer.main)
                             │           │
                    ┌────────▼──┐  ┌─────▼──────────────┐
                    │ PA1 ckpt  │  │ PA2 ckpt            │
                    └───────────┘  └─────────────────────┘
```

### Step 1: 构建基础数据

`build_hier_data.py` 读取原始标注 JSON:
- 输入: `${ANNOTATION_DIR}/*.json` + `${CLIP_DIR_L2}` + `${CLIP_DIR_L3}`
- 处理: L2 用 128s 滑窗（64s 步长）；L3 转为自由分割（无 query）
- 输出: `hier_seg_base_L2_L3/{train,val}.jsonl`

每条记录格式:
```json
{
  "prompt": "<video>\n\n{original prompt text from prompts.py}",
  "answer": "<events>[[10, 35], [40, 72], ...]</events>",
  "videos": ["/path/to/clip.mp4"],
  "problem_type": "temporal_seg_hier_L2",
  "metadata": {"video_id": "...", "duration": 128, ...}
}
```

### Step 2a: PA1 — 直接采样

PA1 从基础数据中按 `problem_type` 分层采样（每层 400 train + 100 val），**不替换 prompt**，保留 `prompts.py` 原始文本。

### Step 2b: PA2 — 替换为 V3 prompt

`prepare_prompt_data.py` 读取基础数据:
1. 提取每条记录中的 `duration` 参数
2. 用 `prompt_variants_v3.py` V2 模板替换 prompt（`.format(duration=N)`）
3. 输出新的 train/val JSONL

---

## 关键路径

| 路径 | 说明 | 控制变量 |
|------|------|---------|
| 标注 JSON | `/m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations/` | `ANNOTATION_DIR` |
| L2 clips | `.../clips/L2/` | `CLIP_DIR_L2` |
| L3 clips | `.../clips/L3/` | `CLIP_DIR_L3` |
| 预处理数据 | `/m2v_intern/.../ablation_data/{EXP_NAME}/` | `ABLATION_DATA_ROOT` |
| Checkpoint | `/m2v_intern/.../hier_seg/ablations/{EXP_NAME}/` | `CHECKPOINT_ROOT` |

---

## V3 Prompt 模板体系（prompt_variants_v3.py）

PA2 使用 V2 变体（Non-CoT + Hard Rules）。完整体系如下:

```
                   No-CoT              Structured CoT（<think> 推理）
                ┌─────────────────┬──────────────────────────────────┐
Boundary-       │  V1  (Baseline) │  V3  CoT-only                    │
criterion       │  MAX_RESP=512   │  MAX_RESP=1024                   │
Hard rules      │  V2  Rules      │  V4  Rules + CoT (Full)          │
+ priors        │  MAX_RESP=512   │  MAX_RESP=1024                   │
                └─────────────────┴──────────────────────────────────┘
```

| Variant | 特点 | PA2 使用 |
|---------|------|---------|
| V1 | 边界判据，无硬规则，无 CoT | |
| **V2** | 边界判据 + 时长先验 + 合并/切分规则 | **← PA2** |
| V3 | V1 + `<think>` 推理 | |
| V4 | V2 + `<think>` 推理 | |

---

## 实验运行

### 1. 批量运行（推荐）

```bash
# 运行 PA1 + PA2
bash local_scripts/hier_seg_ablations/prompt_ablation/run_prompt_ablation.sh

# 仅 PA1
EXPS="PA1" bash local_scripts/hier_seg_ablations/prompt_ablation/run_prompt_ablation.sh

# 快速调试
MAX_STEPS=10 bash local_scripts/hier_seg_ablations/prompt_ablation/run_prompt_ablation.sh
```

### 2. 单次实验

```bash
# PA1: 原始标注 prompt
bash local_scripts/hier_seg_ablations/prompt_ablation/exp_pa1_original.sh

# PA2: V3 边界判据 prompt
bash local_scripts/hier_seg_ablations/prompt_ablation/exp_pa2_v3boundary.sh
```

脚本会自动:
1. 调用 `build_hier_data.py` 从标注 JSON 生成基础 JSONL（若不存在）
2. PA1: 从基础数据采样，保留原始 prompt
3. PA2: 调用 `prepare_prompt_data.py` 替换为 V3 V2 prompt
4. 调用 `launch_train.sh` 启动 `verl.trainer.main` 训练

### 3. 只准备数据（不训练）

```bash
python3 local_scripts/hier_seg_ablations/prompt_ablation/prepare_prompt_data.py \
  --levels L2 L3 \
  --variant V2 \
  --val-per-level 100 \
  --train-per-level 400 \
  --data-root /path/to/base_data \
  --output-dir /path/to/output
```

### 4. 覆盖默认路径

```bash
ABLATION_DATA_ROOT=/your/output/path \
CHECKPOINT_ROOT=/your/ckpt/path \
MAX_STEPS=5 \
bash local_scripts/hier_seg_ablations/prompt_ablation/exp_pa1_original.sh
```

---

## 预期对比分析

| 对比 | 研究问题 |
|------|---------|
| PA1 vs PA2 | 边界判据 + 稀疏约束 + 硬规则是否优于语义描述式 prompt？ |
| PA2 L2 vs PA2 L3 | V3 prompt 在事件级 vs 状态变化级的效果差异 |

预期假设:
- PA2 在 L3（短段、状态变化）上增益更大（稀疏约束帮助避免过碎分割）
- PA1 在 L2（长段、事件级）上可能与 PA2 差距较小（事件边界语义清晰）

---

## 常见问题

**Q: 为什么跳过 L1？**
A: L1 使用 warped frame mapping，58% 的记录 `n_warped_frames > 256`，帧对应不准确。L2+L3 使用秒级时间戳，不受影响。

**Q: 两组实验用的是 CoT 还是 Non-CoT？**
A: **都是 Non-CoT**（MAX_RESPONSE_LEN=512）。PA1 用原始 prompt（无 CoT），PA2 用 V3 V2 变体（Non-CoT + Hard Rules）。CoT 变体（V3/V4, MAX_RESPONSE_LEN=1024）可通过修改 PA2 脚本中的 `--variant` 参数启用。

**Q: 如何切换为 CoT 变体？**
A: 修改 `exp_pa2_v3boundary.sh` 中 `--variant V4`，并在 `common.sh` 前添加 `MAX_RESPONSE_LEN=1024`。
