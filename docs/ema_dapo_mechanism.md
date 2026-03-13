# EMA-GRPO 与 DAPO 机制说明

本文档介绍本项目使用的两种策略优化算法的核心机制，以及如何在多任务视频 RL 训练中结合两者的优势。

---

## 1. GRPO 基础

**GRPO（Group Relative Policy Optimization）** 是 RLHF 训练中替代 PPO+Critic 的轻量方案。核心思想是：**对同一 prompt 生成 n 个回复，用组内相对奖励作为 advantage，省去 Value 网络**。

### 1.1 Advantage 计算

对 prompt $q$ 的 $n$ 次 rollout：

$$A_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$$

其中 $\mu_G, \sigma_G$ 是当前组（同一 prompt）的均值和标准差。

**问题**：若当前 batch 中同一任务的 rollout 全部正确（$\sigma_G \approx 0$），advantage 爆炸；不同任务奖励量纲不同（F1 vs 0/1），归一化效果差。

---

## 2. EMA-GRPO（当前默认算法）

**EMA-GRPO** 用跨 batch 的指数移动平均（EMA）标准差代替即时组内标准差，解决多任务奖励尺度不一致的问题。

### 2.1 核心公式

对每个任务类型（如 `temporal_seg`、`add`、`sort`）维护独立的 EMA 统计量：

$$\hat{m}_1^{(t)} = \alpha \cdot \hat{m}_1^{(t-1)} + (1 - \alpha) \cdot \mathbb{E}_{batch}[r]$$
$$\hat{m}_2^{(t)} = \alpha \cdot \hat{m}_2^{(t-1)} + (1 - \alpha) \cdot \mathbb{E}_{batch}[r^2]$$
$$\hat{\sigma}_{EMA} = \sqrt{\hat{m}_2 - \hat{m}_1^2}$$

Advantage 计算分三步：

1. **组内中心化**：$\hat{r}_i = r_i - \mu_G$（消除 prompt 难度偏差）
2. **EMA 方差归一化**：$A_i = \hat{r}_i / (\hat{\sigma}_{EMA} + \epsilon)$
3. **Guard Rail**：若 $|A_i| > 5$，回退到组内标准差（数值稳定性保障）

### 2.2 关键超参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `ema_decay` | 0.99 | EMA 衰减率，越大记忆越长 |
| `min_std` | 1e-3 | EMA std 下界，防止除零 |
| `guard_abs_max` | 5.0 | Advantage 绝对值上界，触发回退 |

### 2.3 为什么多任务场景必须用 EMA

- `temporal_seg`：F1-IoU，取值范围 $[0, 1]$，方差通常 0.1~0.3
- `add/delete/replace`：二值 0/1，方差通常 0.2~0.5
- `sort`：Jigsaw displacement，取值范围 $[0, 1]$，方差通常 0.15~0.4

若用统一的归一化系数，不同任务的梯度量级相差 2~5 倍，导致训练不稳定。EMA 为每个任务独立维护统计量，**自适应地平衡各任务的梯度贡献**。

---

## 3. DAPO（Dynamic sAmpling Policy Optimization）

DAPO（ByteDance, 2025）在 GRPO 基础上提出 4 项改进，专为大规模 RL 训练设计。

> 参考论文：[DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/pdf/2503.14476)

### 3.1 动态采样（Dynamic Sampling）

**问题**：当某个 prompt 的 n 次 rollout 全部正确（$r_i = 1$）或全部错误（$r_i = 0$）时，组内标准差为 0，advantage 为 0，产生**无效梯度**。

**解法**：在每个训练步动态过滤掉这类"退化组"，持续补充新样本直到 batch 填满：

```
keep group iff:  filter_low < mean(rewards) < filter_high
default:         filter_low=0.01, filter_high=0.99
```

**效果**：仅对"真正有区分度"的 prompt 更新参数，提升训练样本效率。

### 3.2 Clip-Higher（非对称截断）

标准 PPO 使用对称 clip：$[\,1-\varepsilon,\, 1+\varepsilon\,]$。

DAPO 使用**非对称 clip**：允许正向更新更大步长，防止模型探索受限：

$$L_{clip} = -A \cdot \text{clip}(\rho_t,\; 1 - \varepsilon_{low},\; 1 + \varepsilon_{high})$$

其中 $\varepsilon_{low} = 0.2$，$\varepsilon_{high} = 0.3$（默认值）。

### 3.3 Token-Level 策略梯度损失

标准 GRPO 按**序列**平均损失（每条回复贡献相同权重）。DAPO 改为**按 token**平均，长回复（如 CoT 推理链）获得更多梯度信号：

```yaml
loss_avg_mode: token  # DAPO
loss_avg_mode: seq    # 标准 GRPO
```

**对 CoT 训练的意义**：强制模型输出 `<think>` 推理链后，长回复本身是训练目标的一部分，token 级损失能更好地优化每一步推理 token。

### 3.4 移除 KL 惩罚

标准 PPO/GRPO 通常添加 KL 散度项约束策略不偏离 ref policy：

$$L = L_{pg} + \beta \cdot KL(\pi_\theta \| \pi_{ref})$$

DAPO 认为 KL 项**限制了模型探索新格式**（如从"裸答案"进化到 CoT 格式），建议关闭：

```yaml
algorithm.disable_kl: true
```

---

## 4. 本项目方案：EMA-DAPO

本项目将 EMA-GRPO 的多任务归一化优势与 DAPO 的训练效率改进结合，形成 **EMA-DAPO** 组合。

### 4.1 算法组件对比

| 特性 | 原始 GRPO | EMA-GRPO | DAPO | **EMA-DAPO（本项目）** |
|------|-----------|----------|------|----------------------|
| Advantage 基线 | 组内 std | EMA 任务 std | 组内 std | **EMA 任务 std** |
| 动态过滤 | ❌ | ❌ | ✅ | **✅** |
| Clip-Higher | ❌ | ❌ | ✅ | **✅** |
| Token-Level Loss | ❌ | 可选 | ✅ | **✅** |
| 无 KL 惩罚 | ❌ | 可选 | ✅ | **✅** |
| Entropy 正则 | ❌ | ❌ | ❌ | **✅（新增）** |

### 4.2 Entropy 正则（新增）

**问题**：动态过滤导致训练样本分布更窄（仅保留有区分度的 prompt），模型可能过拟合已掌握的题型，导致分布坍塌。

**解法**：在损失中加 Entropy 正则项，鼓励模型维持回复多样性：

$$L_{total} = L_{pg} - \lambda_H \cdot H(\pi_\theta)$$

其中 $H(\pi_\theta) = -\mathbb{E}[\log \pi_\theta]$ 是策略熵（已通过 `-log_probs` 估计）。

```yaml
worker.actor.entropy_coeff: 0.005   # 推荐范围: 0.001 ~ 0.01
```

| `entropy_coeff` | 效果 |
|-----------------|------|
| 0.0 | 无正则，等价于关闭 |
| 0.001 | 轻微探索激励 |
| 0.005 | 推荐（平衡收敛与多样性） |
| 0.01 | 强探索，早期训练防坍塌 |
| > 0.05 | 过强，可能干扰收敛 |

### 4.3 完整配置说明

```yaml
# === EMA-DAPO 核心配置 ===
algorithm:
  adv_estimator: ema_grpo   # EMA 基线：多任务归一化
  online_filtering: true     # DAPO：动态过滤无效组
  filter_key: overall
  filter_low: 0.01           # 剔除全错组
  filter_high: 0.99          # 剔除全对组
  disable_kl: true           # DAPO：移除 KL 约束
  use_kl_loss: false

actor:
  clip_ratio_low: 0.2        # PPO clip 下界
  clip_ratio_high: 0.3       # DAPO Clip-Higher 上界
  clip_ratio_dual: 3.0       # Dual-Clip 安全常数
  loss_avg_mode: token       # DAPO：token 级损失
  entropy_coeff: 0.005       # 新增：entropy 正则
```

### 4.4 与旧脚本的差异

| 参数 | `run_mixed_proxy_training.sh` | `run_mixed_proxy_dapo.sh` |
|------|------------------------------|--------------------------|
| `algorithm.disable_kl` | false | **true** |
| `algorithm.use_kl_loss` | true | **false** |
| `algorithm.online_filtering` | false | **true** |
| `worker.actor.entropy_coeff` | （无此参数）| **0.005** |
| `worker.actor.clip_ratio_high` | （默认 0.3）| **0.3（显式设置）** |
| `worker.actor.loss_avg_mode` | （默认 token）| **token（显式设置）** |

---

## 5. 数据流与训练步骤（EMA-DAPO）

```
每个训练步:
┌─────────────────────────────────────────────────────────────────┐
│ 1. 从 DataLoader 取一个 batch（task_homogeneous_batching）       │
│ 2. vLLM 生成 rollout.n=8 个回复                                  │
│ 3. MixedProxyReward 计算每个回复的 reward                        │
│    - add/delete/replace: 严格要求 <think>+<answer>，否则 0        │
│    - sort: 严格要求 <think>+<answer>，否则 0                      │
│    - temporal_seg: 严格要求 <events>，否则 0                      │
│ 4. 动态过滤（DAPO）:                                             │
│    - 按 UID 分组 → 计算组均值                                    │
│    - 剔除 mean < 0.01 或 mean > 0.99 的组                       │
│    - 若有效 batch 不足，重新采样补充                              │
│ 5. EMA-GRPO Advantage 计算:                                      │
│    - 组内中心化：score_i - mean(group)                           │
│    - EMA 归一化：按 problem_type 独立维护 EMA std                │
│    - Guard Rail：|A| > 5 时回退到组内 std                        │
│ 6. Actor 更新:                                                    │
│    - PPO-Clip-Higher 损失（clip_low=0.2, clip_high=0.3）         │
│    - Token-level 平均                                            │
│    - Entropy 正则：loss -= entropy_coeff * entropy               │
│    - 无 KL 惩罚                                                  │
│ 7. 记录指标：pg_loss, entropy_loss, pg_clipfrac_*               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 调试建议

### 判断动态过滤是否有效

训练日志中关注 `num_try_make_batch` 字段：
- 若持续 = 1，说明过滤不频繁，训练数据充分
- 若频繁 > 3，说明过滤过于激进，可以适当放宽 `filter_low/filter_high`

### 判断 EMA 归一化是否稳定

观察 `ema_std` 相关日志（如有）：
- 早期训练 EMA std 可能偏低（冷启动），这是正常的
- 若 `guard_rail` 触发频繁，说明 EMA 统计还不稳定，需要更多 warmup steps

### 判断 Entropy 正则强度

观察 `actor/entropy_loss` 指标变化：
- 若 entropy 单调下降趋近于 0，说明模型正在坍塌，可适当增大 `entropy_coeff`
- 若 entropy 持续升高，说明正则过强，可减小 `entropy_coeff`

### 判断格式学习是否成功

对 `add/delete/replace/sort` 任务，观察 `reward/format` 指标：
- 若 format 逐渐从 0 上升到 0.5+，说明模型正在学习使用 `<think><answer>` 格式
- 若 format 持续为 0 但 accuracy 也为 0，说明模型陷入了"不输出标签"的局部最优，
  可以适当增大 `entropy_coeff` 或降低 `filter_high`（允许更多全对组参与训练，给格式学习以正向信号）
