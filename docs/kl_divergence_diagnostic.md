# KL 散度异常诊断：忽大忽小忽而为零

> **实验**: `qwen3_vl_mixed_proxy_training` (Qwen3-VL-4B-Instruct, EMA-GRPO, 混合 5 任务)  
> **分析数据**: 前 24 步训练日志 (`experiment_log.jsonl`)  
> **日期**: 2026-03-09

---

## 1. 现象

训练日志中 `actor.kl_loss` 呈现三种异常模式交替出现：

| Step | 任务 | kl_loss | resp_len | advantages | pg_loss | grad_norm |
|------|------|---------|----------|------------|---------|-----------|
| 1 | temporal_seg | 0.000 | 69 | [-0.60, 0.73] | 0.042 | 2.3 |
| 2 | sort | 0.004 | 4.3 | [-1.23, 2.46] | ≈0 | 212 |
| 3 | **add** | **0.000** | **2.0** | [-2.14, 1.29] | 0.021 | 1448 |
| 4 | **delete** | **≈0** | **2.0** | **[0, 0]** | **0.0** | 0.02 |
| 5 | replace | ≈0 | 2.0 | [-0.30, 2.12] | ≈0 | 166 |
| 6 | temporal_seg | 0.003 | 65 | [-0.70, 0.71] | 0.023 | 5.3 |
| 7 | temporal_seg | 0.003 | 77 | [-0.76, 0.69] | 0.025 | 5.2 |
| 8 | temporal_seg | 0.002 | 57 | [-0.97, 0.50] | 0.017 | 5.3 |
| 9 | sort | 0.004 | 4.8 | [-0.35, 0.59] | ≈0 | 48 |
| 10 | **delete** | **≈0** | **2.0** | **[0, 0]** | **0.0** | 0.09 |
| 11 | **replace** | **0.000** | **2.0** | **[0, 0]** | **0.0** | ≈0 |
| 12 | add | 0.003 | 2.0 | [-2.12, 1.27] | ≈0 | 840 |
| 13 | temporal_seg | 0.011 | 55 | [-1.22, 0.99] | -0.004 | 10.4 |
| 14 | temporal_seg | 0.006 | 72 | [-0.71, 1.20] | -0.006 | 15.1 |
| 15 | **sort** | **0.102** | 5.6 | [-1.92, 0.87] | 0.033 | **334** |
| 16 | temporal_seg | 0.012 | 75 | [-0.96, 0.54] | 0.019 | 4.8 |
| 17 | **add** | **≈0** | **2.0** | **[0, 0]** | **0.0** | 6.4 |
| 18 | **replace** | **0.111** | **2.0** | [-0.91, 1.52] | 0.005 | **3808** |
| 19 | delete | 0.001 | 2.0 | [-1.00, 1.00] | -0.028 | 692 |
| 20 | temporal_seg | 0.017 | 49 | [-1.94, 0.76] | -0.039 | 22.4 |
| 21 | temporal_seg | 0.022 | 49 | [-0.95, 1.22] | -0.024 | 9.5 |
| 22 | temporal_seg | 0.022 | 55 | [-0.58, 0.85] | 0.030 | 8.8 |
| 23 | **sort** | **0.248** | 5.3 | [-0.61, 0.79] | 0.004 | **336** |
| 24 | **delete** | 0.001 | **2.0** | **[0, 0]** | **0.0** | 5.7 |

可以清晰地看到三种模式：

```
Pattern A: KL ≈ 0, advantage = [0, 0]     ← 选择题采样无差异
Pattern B: KL 平稳增长 (0.003 → 0.022)     ← temporal_seg 正常训练
Pattern C: KL 跳变 (0.1 → 0.25)            ← 累积漂移突然暴露
```

---

## 2. 根因分析

### 根因 1: 选择题 GRPO 熵坍缩（Pattern A — KL 恒为 0）

**因果链**:

```
选择题 prompt 要求 "Output your answer as a single letter"
        ↓
模型回复仅 2 token (字母 + EOS)，如 "C\n"
        ↓
temperature=0.7 + response_length=2 → 采样几乎无随机性
        ↓
rollout.n=8 次采样结果完全一致 (全部输出 "C\n")
        ↓
GRPO 组内: r₁ = r₂ = ... = r₈ → μ_g = r₁, σ_g = 0
        ↓
advantage = (r_i - μ_g) / σ_g = 0 / 0 → 0 (mean centering 后已为 0)
        ↓
pg_loss = 0, 梯度 = 0, 策略不更新
        ↓
当前策略 = 参考策略 (因为没更新过)
        ↓
KL = 0
```

**日志证据**:

- Step 4 (delete): reward.overall=1.0, 但 advantages=\[0, 0\]。说明 batch 里 4 个 prompt × 8 rollouts，**每个 prompt 的 8 次回复完全一样**。
- Step 11 (replace): reward=1.0, advantages=\[0, 0\], pg_loss=0, grad_norm=2.4e-5。模型完全自信，没有学习信号。
- Step 17 (add): reward=0.75（4 个 prompt 中 3 个全对 1 个全错），advantages=\[0, 0\]。

**核心问题**: GRPO 依赖 **组内采样多样性** 来产生梯度信号。当 response 仅 2 个 token 时，模型输出是近乎确定性的，GRPO 完全失效。

### 根因 2: 任务间策略漂移累积（Pattern C — KL 跳变）

**因果链**:

```
temporal_seg 任务连续多步产生有效梯度 (Steps 6→7→8, 13→14, 20→21→22)
        ↓
每步更新全局修改了模型权重
        ↓
但选择题因 advantage=0 从不更新自己的策略
        ↓
模型在 sort/choice token 分布上的偏移被「顺带」制造，但无法被纠正
        ↓
偏移单向累积
        ↓
当 sort/replace batch 终于有有效梯度时
        ↓
KL = D_KL(π_current ‖ π_ref) 暴露出巨大跳变
```

**日志证据**:

```
Steps 6→7→8 (temporal_seg): KL 稳步 0.003 → 0.003 → 0.002
Steps 13→14  (temporal_seg): KL 涨到 0.011 → 0.006

Step 15 (sort):    KL 突然跳到 0.102 ← 中间 6 步 temporal_seg 的累积漂移
Step 18 (replace): KL 突然跳到 0.111 ← 被 step 16 的 temporal_seg 更新放大
Step 23 (sort):    KL 飙到 0.248   ← 3 步 temporal_seg (20/21/22) 累积
```

**关键观察**: KL 的跳变总是发生在 **temporal_seg 连续训练多步之后的非 temporal_seg batch** 上。这不是随机抖动，而是因果性的漂移积累。

### 根因 3: 异常梯度幅度（间接后果）

选择题偶尔有方差时（不是所有 prompt 都全对/全错），产生的 advantage 很大（因为 EMA std 被 temporal_seg 主导设置，对选择题的 {0,1} 二元 reward 产生过大缩放），导致巨大梯度：

| Step | 任务 | grad_norm | 说明 |
|------|------|-----------|------|
| 3 | add | **1448** | advantage 有值但 EMA std 初始不稳 |
| 12 | add | **840** | 类似 |
| 15 | sort | **334** | 漂移累积后 KL 跳变 |
| 18 | replace | **3808** | 最大跳变点 |
| 19 | delete | 692 | 连带效应 |

虽然 `max_grad_norm=1.0` 的 clipping 可以防止参数爆炸，但 clipped 后的有效学习率：

$$\text{effective\_lr} = \text{lr} \times \frac{\text{max\_grad\_norm}}{\text{grad\_norm}} = 8 \times 10^{-7} \times \frac{1}{3808} \approx 2.1 \times 10^{-10}$$

这意味着 step 18 的 actor 更新几乎等于 **零**，整步训练资源浪费。

---

## 3. 根本问题总结

```
┌────────────────────────────────────────────────────────────────┐
│ 选择题 response 仅 2 token → GRPO 采样无差异 → 无梯度信号     │
│     ↕                                                          │
│ temporal_seg 正常训练 → 全局权重被单方向更新                    │
│     ↕                                                          │
│ 选择题 KL 被动漂移 + 无法自我纠正 → KL 忽大忽小忽为零          │
│     ↕                                                          │
│ 偶尔有梯度时幅度巨大 → grad_norm 达到数千 → clip 后等于没学    │
└────────────────────────────────────────────────────────────────┘
```

一句话概括：**选择题太短了，GRPO 没有探索空间**。

---

## 4. 修复方案

### 方案 A: Prompt 重设计（已实施 ✅）

**核心改动**: 将选择题/排序题的输出指令从

```
Output your answer as a single letter (e.g., A, B, C, D).
```

改为 Chain-of-Thought 格式：

```
First, carefully observe the actions and visual content in each Context Video...
Think step by step inside <think> </think> tags, then provide your final
answer (a single letter A, B, C, or D) inside <answer> </answer> tags.
```

**预期效果**:

| 指标 | 改前 | 改后预期 |
|------|------|----------|
| 选择题 response_length | 2 tokens | 50-200 tokens |
| 8 次采样一致率 | ~100% | < 30% |
| advantage=0 比例 | ~70% 的选择题 batch | < 10% |
| KL 跳变幅度 | 0 → 0.25 | 平滑增长 |

**实施文件**:

- `proxy_data/redesign_prompts.py` — Prompt 重写脚本
- `proxy_data/mixed_train_cot.jsonl` — 新训练数据（9446 样本）
- `verl/reward_function/mixed_proxy_reward.py` — 支持从 `<answer>` 标签提取答案
- `local_scripts/run_mixed_proxy_training.sh` — `MAX_RESPONSE_LEN` 从 512 调至 2048

### 方案 B: 采样温度分离（可选附加）

对不同任务使用不同的 rollout 温度：

```yaml
# 选择题用更高温度增加采样多样性
temperature_by_task:
  add: 1.0
  delete: 1.0
  replace: 1.0
  sort: 0.9
  temporal_seg: 0.7  # 长回复本身就有足够多样性
```

> 注: 当前框架不支持 per-task temperature，需要修改 vLLM rollout 代码。方案 A 已经足够解决问题，此方案作为进一步优化。

### 方案 C: 参数层面保护（可选附加）

1. **EMA warmup**: 在前 N 步用较大的 `min_std`（如 0.5），防止 EMA std 初始值过小导致 advantage 放大
2. **per-task KL tracking**: 分任务记录 KL，设置告警阈值（如 KL > 0.1 时自动降低该任务的学习率）
3. **adaptive rollout.n**: 如果检测到某任务 advantage 全为 0，自动增加采样次数到 16 或 32

---

## 5. 验证清单

修复后重新训练时，需要关注的指标：

- [ ] 选择题 `response_length` > 30（不再是 2）
- [ ] 选择题 `advantages` 不全为 0（至少 min ≠ max）
- [ ] `kl_loss` 随步数平滑增长，不再出现 0 → 0.1 的跳变
- [ ] `grad_norm` < 100（不再出现 1000+ 的爆发）
- [ ] `reward/format` > 0.5（模型学会使用 `<answer>` 标签）
- [ ] 各任务 reward 均有提升趋势

---

## 附录: 数据提取脚本

```python
# 从 experiment_log.jsonl 提取 KL 分析数据
import json

with open("checkpoints/qwen3_vl_mixed_proxy_training/experiment_log.jsonl") as f:
    for line in f:
        d = json.loads(line)
        step = d["step"]
        kl = d["actor"]["kl_loss"]
        pg = d["actor"]["pg_loss"]
        gn = d["actor"]["grad_norm"]
        rl = d["response_length"]["mean"]
        adv_min = d["critic"]["advantages"]["min"]
        adv_max = d["critic"]["advantages"]["max"]
        
        # 确定任务类型
        task = "?"
        for key in d["reward"]:
            if key not in ("overall", "format", "accuracy"):
                task = key
        
        print(f"Step {step:2d} | {task:12s} | KL={kl:.6f} | "
              f"resp={rl:5.1f} | adv=[{adv_min:.2f}, {adv_max:.2f}] | "
              f"pg={pg:.4f} | grad={gn:.1f}")
```
