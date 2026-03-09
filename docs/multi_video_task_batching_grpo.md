# 多视频处理 · 任务分 Batch · EMA-GRPO 算法详解

> 本文档详细讲解 EasyR1 框架中**多视频数据处理**、**任务同质化 Batch 采样**和 **EMA-GRPO 优势估计**三大核心机制的设计原理与代码实现。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [多视频处理流水线](#2-多视频处理流水线)
   - 2.1 [数据格式](#21-数据格式)
   - 2.2 [Dataset 阶段：编码 prompt](#22-dataset-阶段编码-prompt)
   - 2.3 [Rollout 阶段：vLLM 推理](#23-rollout-阶段vllm-推理)
   - 2.4 [FSDP Worker 阶段：重新处理视频](#24-fsdp-worker-阶段重新处理视频)
   - 2.5 [为什么要「逐视频处理再 cat」？](#25-为什么要逐视频处理再-cat)
   - 2.6 [多视频 max_frames 分配](#26-多视频-max_frames-分配)
3. [任务同质化 Batch 采样](#3-任务同质化-batch-采样)
   - 3.1 [问题：混合任务 Batch 的梯度冲突](#31-问题混合任务-batch-的梯度冲突)
   - 3.2 [TaskHomogeneousBatchSampler 设计](#32-taskhomogeneousbatchsampler-设计)
   - 3.3 [加权轮转交错](#33-加权轮转交错)
   - 3.4 [与 DataLoader 的集成](#34-与-dataloader-的集成)
4. [EMA-GRPO 优势估计算法](#4-ema-grpo-优势估计算法)
   - 4.1 [GRPO 回顾](#41-grpo-回顾)
   - 4.2 [GRPO 在多任务场景的问题](#42-grpo-在多任务场景的问题)
   - 4.3 [EMA-GRPO 算法](#43-ema-grpo-算法)
   - 4.4 [Guard Rail 机制](#44-guard-rail-机制)
5. [端到端训练流程](#5-端到端训练流程)
6. [Reward 函数设计](#6-reward-函数设计)

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                      Driver (ray_trainer.py)                    │
│                                                                 │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌─────────┐ │
│  │DataLoader│───>│  Rollout   │───>│  Reward  │───>│ EMA-GRPO│ │
│  │ (TaskBatch│    │  (vLLM)   │    │ Function │    │Advantage│ │
│  │ Sampler) │    │            │    │          │    │         │ │
│  └──────────┘    └────────────┘    └──────────┘    └────┬────┘ │
│                                                         │      │
│                  ┌────────────┐    ┌──────────┐         │      │
│                  │Actor Update│<───│ Critic   │<────────┘      │
│                  │  (FSDP)    │    │ (FSDP)   │                │
│                  └────────────┘    └──────────┘                │
└─────────────────────────────────────────────────────────────────┘
```

关键数据流：

1. **DataLoader** 从混合数据集中，通过 `TaskHomogeneousBatchSampler` 取出**同一任务**的 batch
2. **Dataset `__getitem__`** 读取多个视频 → 编码为 token → 存储 `multi_modal_data`
3. **Rollout (vLLM)** 利用 `multi_modal_data` 重新加载视频进行生成
4. **FSDP Worker** 再次利用 `multi_modal_data` 处理视频计算 log_probs / values
5. **Reward Function** 根据 `problem_type` 分派到不同 reward 计算器
6. **EMA-GRPO** 按 `problem_type` 追踪各任务的 EMA 标准差，做归一化后计算优势

---

## 2. 多视频处理流水线

### 2.1 数据格式

每个样本的 JSONL 结构：

```json
{
  "messages": [{"role": "user", "content": "Video1. <video>\nVideo2. <video>\n...排序这些视频"}],
  "prompt": "...",
  "answer": "13245",
  "videos": [
    "/path/to/clip1.mp4",
    "/path/to/clip2.mp4",
    "/path/to/clip3.mp4"
  ],
  "problem_type": "sort",
  "data_type": "video"
}
```

其中 `<video>` 占位符的数量**必须**与 `videos` 数组长度一致。

### 2.2 Dataset 阶段：编码 prompt

**文件**: `verl/utils/dataset.py` → `RLHFDataset.__getitem__`

```
输入: videos = ["clip1.mp4", "clip2.mp4", "clip3.mp4"]
      prompt = "Video1. <video>\nVideo2. <video>\nVideo3. <video>\n排序..."

                    ┌──────────────┐
                    │ _build_msg   │  将 <video> 转成 {"type": "video"}
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ apply_chat_  │  得到含 <|vision_start|>...<|vision_end|>
                    │ template()   │  占位符的完整 prompt 文本
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │  逐视频 process_video() │
              │  max_frames_per_video   │
              │  = max_frames // N      │
              │                         │
              │  clip1 → (T1,C,H,W)     │
              │  clip2 → (T2,C,H,W)     │
              │  clip3 → (T3,C,H,W)     │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  processor(             │  Qwen3VLProcessor
              │    text=[prompt],       │  将视频帧编码为 pixel_values
              │    videos=all_frames,   │  和 video_grid_thw
              │    video_metadata=      │
              │      all_metadatas      │
              │  )                      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  输出:                   │
              │  - input_ids (prompt)    │
              │  - attention_mask        │
              │  - multi_modal_data = {  │  ← 存储视频路径和处理参数
              │      "videos": [...],   │     供后续阶段重新处理
              │      "max_frames": X,   │     注意: 这里是 per-video 值
              │      ...                │
              │    }                    │
              └─────────────────────────┘
```

**关键设计点**：

- `multi_modal_data` 仅存储**视频路径**和处理参数（而非像素数据），因为它需要跨进程序列化
- `max_frames` 存储的是 **per-video** 的值（`self.max_frames // n_videos`），确保后续阶段一致

### 2.3 Rollout 阶段：vLLM 推理

**文件**: `verl/workers/rollout/vllm_rollout_spmd.py`

```
multi_modal_data (包含视频路径和参数)
         │
         ▼
_process_multi_modal_data()
  ├─ 对每个视频: process_video(video, **kwargs)
  │   kwargs = {max_frames, min_pixels, max_pixels, video_fps}
  │   从 multi_modal_data 直接传递
  │
  └─ 组装 mm_data + mm_kwargs
         │
         ▼
vLLM engine.generate(prompt, multi_modal_data=mm_data)
         │
         ▼
模型生成 response tokens
```

vLLM 内部有自己的视频编码器，会根据传入的像素数据推理。

### 2.4 FSDP Worker 阶段：重新处理视频

**文件**: `verl/workers/fsdp_workers.py` → `_process_multi_modal_inputs`

这个阶段发生在：
- `compute_log_probs`（计算旧策略的 log prob）
- `compute_ref_log_probs`（计算参考策略的 log prob）
- `update_actor` / `update_critic`（策略/价值网络更新）

```
multi_modal_data (来自 Dataset 的视频路径)
         │
         ▼
  ┌──────────────────────────────────┐
  │  对每个样本:                      │
  │    对每个视频:                    │
  │      process_video() → tensor    │
  │                                  │
  │    逐视频调用 image_processor:    │
  │      video1 → r1 {pixel_values,  │
  │                    video_grid_thw}│
  │      video2 → r2 {pixel_values,  │
  │                    video_grid_thw}│
  │      video3 → r3 {...}           │
  │                                  │
  │    torch.cat([r1, r2, r3], dim=0)│  ← 关键！不是 torch.stack
  └──────────────────────────────────┘
```

### 2.5 为什么要「逐视频处理再 cat」？

这是解决多视频处理中最关键的 bug。

#### ❌ 旧代码（一次性传所有视频）

```python
# 旧代码：把所有视频一起传
multi_modal_inputs = dict(
    self.processor.image_processor(
        images=None,
        videos=videos,     # [video1(4帧), video2(52帧)]
        return_tensors="pt",
        do_resize=False,
    )
)
```

`image_processor` 内部的处理流程：

```
videos = [tensor(4, C, H, W), tensor(52, C, H, W)]
                    │
                    ▼
group_images_by_shape()
  按 (C, H, W) 分组:
  组1 (3, 160, 288):
    - tensor(4, 3, 160, 288)    ← 来自 video1
    - tensor(52, 3, 160, 288)   ← 来自 video2
                    │
                    ▼
torch.stack(组1的所有tensor, dim=0)
  stack([shape(4,...), shape(52,...)])
  💥 RuntimeError: stack expects each tensor to be equal size,
     but got [4, 3, 160, 288] at entry 0 and [52, 3, 160, 288] at entry 1
```

**根本原因**：`torch.stack` 要求**所有维度**完全一致，不同视频帧数不同（4 vs 52），stack 失败。

#### ✅ 新代码（逐视频处理后 cat）

```python
# 新代码：每个视频单独处理
per_video_results = []
for single_video in videos:
    r = dict(
        self.processor.image_processor(
            images=None,
            videos=[single_video],      # 只传 1 个视频
            return_tensors="pt",
            do_resize=False,
        )
    )
    per_video_results.append(r)

# 合并结果
multi_modal_inputs = {}
for key in per_video_results[0]:
    tensors = [r[key] for r in per_video_results]
    if isinstance(tensors[0], torch.Tensor):
        multi_modal_inputs[key] = torch.cat(tensors, dim=0)  # cat, 不是 stack!
```

每次只传 1 个视频，processor 内部 stack 就不会有维度冲突：

```
video1 单独处理:
  group_images_by_shape → 只有 1 个 tensor(4, C, H, W)
  torch.stack([tensor(4, C, H, W)]) → tensor(4, C, H, W)  ✅

video2 单独处理:
  group_images_by_shape → 只有 1 个 tensor(52, C, H, W)
  torch.stack([tensor(52, C, H, W)]) → tensor(52, C, H, W)  ✅

最后 torch.cat:
  cat([tensor(4, C, H, W), tensor(52, C, H, W)], dim=0)
  → tensor(56, C, H, W)  ✅ cat 只要求 dim=0 以外的维度一致
```

**总结核心区别**：

| 操作 | 要求 | 是否成功 |
|------|------|---------|
| `torch.stack([shape(4,...), shape(52,...)])` | **所有维度**完全一致 | ❌ 帧数不同 |
| `torch.cat([shape(4,...), shape(52,...)], dim=0)` | 仅要求 **dim≠0** 一致 | ✅ spatial 一致即可 |

### 2.6 多视频 max_frames 分配

**问题**：如果 `max_frames=256`，一个样本有 8 个视频，每个视频都用 256 帧 → 总计 2048 帧 → OOM。

**解决方案**：`dataset.py` 中按视频数量均匀分配帧预算：

```python
n_videos = len(videos)
if n_videos > 1:
    max_frames_per_video = max(1, self.max_frames // n_videos)
else:
    max_frames_per_video = self.max_frames
```

各任务实际帧预算示例（`max_frames=256`）：

| 任务 | 视频数 | 每视频帧数 | 总帧数 |
|------|--------|-----------|--------|
| temporal_seg | 1 | 256 | 256 |
| sort | 3~5 | 51~85 | 153~256 |
| add / replace | 6~8 | 32~42 | 192~256 |
| delete | 4~8 | 32~64 | 128~256 |

**关键**：`multi_modal_data["max_frames"]` 存储的是 `max_frames_per_video`（而非原始 256），确保后续 rollout / FSDP worker 重新处理视频时使用相同的帧预算。

---

## 3. 任务同质化 Batch 采样

### 3.1 问题：混合任务 Batch 的梯度冲突

在混合训练中，如果一个 batch 同时包含：
- `sort` 任务（4 个视频，回复是数字序列）
- `temporal_seg` 任务（1 个视频，回复是事件标签）

会出现两个问题：

1. **梯度方向冲突**：不同任务可能需要模型往相反方向更新
2. **优势估计污染**：GRPO 使用 batch 内统计量（均值/标准差）归一化，不同任务的 reward 尺度差异大（如 temporal_seg 的 F1-IoU ∈ [0,1] vs 选择题 ∈ {0, 1}），混在一起会导致归一化失效

### 3.2 TaskHomogeneousBatchSampler 设计

**文件**: `verl/utils/task_sampler.py`

核心保证：**每个 batch 内的样本一定来自同一个任务**。

```
数据集 (9795 样本)
  │
  ▼
按 problem_type 分桶:
  ┌─────────────┬──────┐
  │ temporal_seg │ 4463 │
  │ add          │ 1333 │
  │ delete       │ 1333 │
  │ replace      │ 1333 │
  │ sort         │ 1333 │
  └─────────────┴──────┘
  │
  ▼
每桶独立 shuffle
  │
  ▼
按权重计算各桶 batch 数
  temporal_seg: 40% → 111 batches
  add:          15% → 42 batches
  delete:       15% → 42 batches
  replace:      15% → 42 batches
  sort:         15% → 42 batches
  │
  ▼
交错排列 → 最终 batch 序列
```

### 3.3 加权轮转交错

使用 **fractional stride** 方法将不同任务的 batch 均匀穿插（避免连续多个 batch 都是同一任务）：

```python
def _interleave_batches(self, task_batches, rng):
    total = sum(len(b) for b in task_batches.values())
    positioned = []
    for t, batches in task_batches.items():
        stride = total / len(batches)
        for i, batch in enumerate(batches):
            pos = i * stride + rng.uniform(0, stride * 0.3)
            positioned.append((pos, batch))
    positioned.sort(key=lambda x: x[0])
    return [batch for _, batch in positioned]
```

效果示意（T=temporal_seg, A=add, S=sort, D=delete, R=replace）：

```
Epoch batch 序列:
T T A T S T D T R T T A T S T D T R T T A ...
^───────────────────^
  temporal_seg 占40%，
  约每 2.5 个 batch 出现 1 次其他任务
```

### 3.4 与 DataLoader 的集成

**文件**: `verl/trainer/data_loader.py`

```python
if config.task_homogeneous_batching:
    task_weights = dict(config.task_weights) if config.task_weights else None

    batch_sampler = TaskHomogeneousBatchSampler(
        dataset=train_dataset,
        batch_size=train_batch_size,
        task_key=config.task_key,         # "problem_type"
        task_weights=task_weights,
        seed=config.seed,
        drop_last=True,
    )
    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,      # ← 用 batch_sampler，不用 sampler
        ...
    )
```

注意：使用 `batch_sampler` 时，`shuffle`/`sampler` 参数不能同时指定。

---

## 4. EMA-GRPO 优势估计算法

### 4.1 GRPO 回顾

标准 GRPO（Group Relative Policy Optimization）：

对每个 prompt，生成 $n$ 个候选回复（`rollout.n`），计算各自的 reward $r_1, r_2, ..., r_n$，然后组内归一化：

$$
\hat{A}_i = \frac{r_i - \mu_g}{\sigma_g + \epsilon}
$$

其中 $\mu_g = \text{mean}(r_1, ..., r_n)$，$\sigma_g = \text{std}(r_1, ..., r_n)$。

### 4.2 GRPO 在多任务场景的问题

当不同任务的 reward 尺度差异大时：

| 任务 | reward 范围 | 组内 std |
|------|-------------|---------|
| temporal_seg | [0, 1] 连续 | ~0.15 |
| sort (jigsaw) | [0, 1] 连续 | ~0.20 |
| add/delete/replace | {0, 1} 二元 | ~0.50 |

**问题**：temporal_seg 的 std=0.15 远小于选择题 std=0.50，用各自的组内 std 归一化后，temporal_seg 的优势被放大了 ~3x，导致梯度被 temporal_seg 主导。

### 4.3 EMA-GRPO 算法

**文件**: `verl/trainer/core_algos.py` → `compute_ema_grpo_outcome_advantage`

EMA-GRPO 的核心思想：**保留组内均值中心化，但用「任务级别 EMA 标准差」替代「组内标准差」做缩放**。

#### 算法步骤

**Step 1: 组内均值中心化（与 GRPO 相同）**

$$
\tilde{r}_i = r_i - \mu_g
$$

仍然用同一 prompt 的多个回复做 mean centering，减少 intra-group 方差。

**Step 2: 用当前 batch 更新 EMA 统计量**

对每个任务 $t$ 维护 EMA 一阶矩和二阶矩：

$$
m_1^{(t)} \leftarrow \alpha \cdot m_1^{(t)} + (1-\alpha) \cdot \bar{r}^{(t)}_{\text{batch}}
$$

$$
m_2^{(t)} \leftarrow \alpha \cdot m_2^{(t)} + (1-\alpha) \cdot \overline{r^2}^{(t)}_{\text{batch}}
$$

$$
\sigma_{\text{EMA}}^{(t)} = \sqrt{\max(m_2^{(t)} - (m_1^{(t)})^2, \ \sigma_{\min}^2)}
$$

其中 $\alpha = 0.99$ 为 EMA 衰减系数。

**Step 3: 用 EMA std 做缩放**

$$
\hat{A}_i = \frac{\tilde{r}_i}{\sigma_{\text{EMA}}^{(t)} + \epsilon}
$$

**直觉**：这相当于所有任务共享一个"全局视角"的缩放尺度，但每个任务有自己的 EMA 追踪器，所以不会被其他任务的 reward 尺度干扰。

#### 代码核心逻辑

```python
# Step 1: 组内 mean-centering
for gid, pos_list in gid_to_pos.items():
    g_mean = scores[pos_list].mean()
    centered[pos_list] = scores[pos_list] - g_mean

# Step 2: 更新 EMA（先更新，再读取；让统计量包含当前 batch）
for key, pos_list in task_to_pos.items():
    _EMA_STD_TRACKER.update_with_batch_scores(key, scores[pos_list])

# Step 3: 用 EMA std 缩放
for key, task_pos in task_to_pos.items():
    task_std = _EMA_STD_TRACKER.get_std(key)
    scaled[task_pos] = centered[task_pos] / (task_std + eps)
```

### 4.4 Guard Rail 机制

EMA std 可能因为初始化不稳定导致过小（放大效应），所以有一个**保护机制**：

```python
# 如果缩放后任何值超过 ±guard_abs_max（默认 5），
# 该 group 回退到 GRPO 的组内 std
if torch.any(torch.abs(tmp) > guard_abs_max):
    g_std = torch.std(scores[gpos], unbiased=False).item()
    scaled[gpos] = centered[gpos] / (g_std + eps)
```

流程图：

```
                      centered / task_std
                           │
                    ┌──────▼──────┐
                    │ |value| > 5 ?│
                    └──────┬──────┘
                     Yes   │   No
                     │     │     │
              ┌──────▼──┐  │  ┌──▼────────┐
              │ 回退到   │  │  │ 保留       │
              │ group std│  │  │ task EMA   │
              │ 缩放     │  │  │ std 缩放   │
              └─────────┘  │  └───────────┘
                           │
                           ▼
                      advantages
```

---

## 5. 端到端训练流程

一个完整 step 的执行过程：

```
Step N
│
├─ 1. DataLoader.next()
│     TaskHomogeneousBatchSampler → 取出同一任务的 batch
│     RLHFDataset.__getitem__ × batch_size:
│       - 逐视频 process_video (max_frames / n_videos per video)
│       - processor 编码 → input_ids + multi_modal_data
│
├─ 2. Rollout (vLLM)
│     prepare_rollout_engine() → FSDP weights → vLLM
│     generate_sequences():
│       - 从 multi_modal_data 重新加载视频
│       - vLLM 生成 n 个候选回复 (rollout.n=8)
│     release_rollout_engine() → vLLM offload
│
├─ 3. Balance batch
│     按 token 数量重排样本到各 DP rank，平衡负载
│
├─ 4. Reward 计算 (异步)
│     reward_fn.compute_reward.remote(batch)
│       - mixed_proxy_reward.compute_score()
│       - 按 problem_type 分派:
│         add/delete/replace → _choice_reward (精确匹配)
│         sort → _sort_reward (jigsaw displacement)
│         temporal_seg → _temporal_seg_reward (F1-IoU)
│
├─ 5. Old log_probs 计算
│     FSDP Worker._process_multi_modal_inputs():
│       - 从 multi_modal_data 重新处理视频
│       - 逐视频调用 image_processor → torch.cat
│     actor.compute_log_prob(data)
│
├─ 6. Ref log_probs 计算 (if use_reference_policy)
│     同 Step 5，但用 ref 模型
│
├─ 7. Advantage 计算
│     reward_tensor → token_level_scores
│     compute_advantage(adv_estimator="ema_grpo"):
│       - 按 problem_type 分任务
│       - 组内 mean centering
│       - 更新 EMA std
│       - task-level std 缩放 (带 guard rail)
│       → advantages, returns
│
├─ 8. Actor 更新
│     PPO clipped objective + advantages
│     梯度更新 (micro_batch_size 控制显存)
│
├─ 9. 保存 rollout 日志
│     _save_train_rollouts():
│       - 记录 step, uid, problem_type, prompt, response, reward
│       - 按任务统计 reward 均值 → wandb/tensorboard
│
└─ 10. Checkpoint (按 save_freq)
```

---

## 6. Reward 函数设计

**文件**: `verl/reward_function/mixed_proxy_reward.py`

### 统一入口

```python
def compute_score(reward_inputs: List[Dict]) -> List[Dict[str, float]]:
    # 根据 problem_type 分派
    dispatch = {
        "add":          _choice_reward,
        "delete":       _choice_reward,
        "replace":      _choice_reward,
        "sort":         _sort_reward,
        "temporal_seg": _temporal_seg_reward,
    }
```

### 各任务 Reward

#### 选择题 (add / delete / replace)

```
GT: "C" (单个字母)
Model output: "The answer is C"

提取: 找到最后出现的大写字母 → "C"
匹配: pred == gt → reward = 1.0
不匹配 → reward = 0.0
无法解析 → reward = 0.0
```

#### 排序题 (sort)

```
GT: "13245" (数字序列, 1-索引)
Model output: "The correct order is 13245"

解析: 提取所有数字 → [1, 3, 2, 4, 5]
计算 Jigsaw Displacement:
  E_jigsaw = Σ |pos_pred(k) - pos_gt(k)|
  E_max    = Σ |i - (n-1-i)|  (完全逆序)
  R_jigsaw = 1 - E_jigsaw / E_max

完全正确 → 1.0
部分正确 → R_jigsaw ∈ (0, 1)
无法解析 → 0.0
```

#### 时序分割 (temporal_seg)

```
GT: "<events>\n[0, 12]\n[15, 30]\n</events>"
Model output: "<events>\n[0, 10]\n[14, 32]\n</events>"

检查: 必须有 <events>...</events> 标签，否则 = 0.0
解析: 提取时间区间对
计算: F1-IoU (precision × recall 的调和平均，用 IoU 匹配)
```

### 核心原则

**格式不对就不给奖励**：所有任务的 `format` 字段恒为 0.0。如果模型输出无法解析，`overall = 0.0`，不存在"格式分"。这迫使模型学会正确的输出格式。

---

## 附录：配置示例

```yaml
# 数据配置
data:
  task_homogeneous_batching: true
  task_weights:
    temporal_seg: 0.40
    add: 0.15
    delete: 0.15
    replace: 0.15
    sort: 0.15
  task_key: "problem_type"
  max_frames: 256

# 算法配置
algorithm:
  adv_estimator: ema_grpo

# Rollout 配置
rollout:
  n: 8  # 每个 prompt 生成 8 个候选

# Reward 配置
reward_function: "verl/reward_function/mixed_proxy_reward.py:compute_score"
```
