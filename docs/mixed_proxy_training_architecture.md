# `run_mixed_proxy_training.sh` 与当前多任务训练框架解读

## 1. 这份脚本在做什么

`local_scripts/run_mixed_proxy_training.sh` 的目标是启动一个“多任务混训”的 EasyR1 训练任务，把两类数据放进同一套 RL 训练链路里：

- 代理任务：`add` / `delete` / `replace` / `sort`
- 时序分割任务：`temporal_seg`

脚本顶部注释已经把它的设计意图写得很清楚，核心是四件事：

1. `task_homogeneous_batching=true`
2. `task_weights=...`
3. `mixed_proxy_reward.py:compute_score`
4. `algorithm.adv_estimator=ema_grpo`

这四项分别对应：

- 采样阶段：一个 batch 只放同一种任务
- 任务配比阶段：不同任务按指定比例出现
- reward 阶段：用一个统一 reward 入口按 `problem_type` 分发
- advantage 阶段：按任务维度维护各自的 EMA 标准差

## 2. 脚本参数的真实含义

### 2.1 数据与任务定义

脚本中最关键的数据参数是：

```bash
TRAIN_FILE="proxy_data/mixed_train.jsonl"
TEST_FILE="proxy_data/youcook2_val_small.jsonl"
TASK_WEIGHTS='{"temporal_seg":0.40,"add":0.15,"delete":0.15,"replace":0.15,"sort":0.15}'
REWARD_FUNCTION="verl/reward_function/mixed_proxy_reward.py:compute_score"
```

含义：

- `TRAIN_FILE` 是已经混合好的多任务训练集
- `TASK_WEIGHTS` 规定采样比例，不是 loss 权重
- `REWARD_FUNCTION` 指向统一 reward 分发函数

`proxy_data/merge_datasets.py` 负责构造这个混合训练集：

- proxy 数据优先从 `metadata.task_type` 写入 `problem_type`
- 时序分割数据默认写成 `problem_type="temporal_seg"`

所以最终训练框架真正依赖的任务标签字段是 `problem_type`。

### 2.2 采样策略

脚本里：

```bash
data.task_homogeneous_batching=true
data.task_weights="${TASK_WEIGHTS}"
data.task_key="problem_type"
```

意思是：

- dataloader 不再随机混采
- 而是先按 `problem_type` 分桶
- 每次只从某一个任务桶里取满一个 batch
- 再按 `task_weights` 决定不同任务 batch 在一个 epoch 中出现的频率

这解决的是“不同任务 reward 尺度和输出格式差异很大，混在同一个 batch 里不稳定”的问题。

### 2.3 GRPO 相关设置

脚本里：

```bash
ADV_ESTIMATOR=ema_grpo
ROLLOUT_N=8
```

含义：

- 每个 prompt 生成 `n=8` 个候选回复
- 用同一个 prompt 的 8 个回复形成 GRPO group
- 再在 `ema_grpo` 中，先按 group 做均值中心化，再按任务做标准差缩放

如果把这里改成普通 `grpo`，那么多任务区分只剩下 reward 分发，不再有“按任务维护各自标准差”的逻辑。

### 2.4 KL 与学习率

脚本还显式打开了 ref policy + KL loss：

```bash
algorithm.disable_kl=false
algorithm.use_kl_loss=true
algorithm.kl_penalty=low_var_kl
algorithm.kl_coef=0.1
```

也就是说：

- reward 本身先由任务 reward 给出
- actor 更新时再额外加 KL 约束
- 这不是多任务特有逻辑，但对混训很重要，因为多任务 reward 分布更容易造成策略发散

## 3. 当前训练框架里，多任务是怎么被处理的

可以把当前实现理解成下面这条链：

```text
mixed_train.jsonl
  -> RLHFDataset
  -> TaskHomogeneousBatchSampler
  -> RayPPOTrainer._make_batch_data()
  -> rollout 生成 n 个回复
  -> reward manager 传入 problem_type / data_type
  -> mixed_proxy_reward 按任务分发
  -> compute_advantage()
  -> ema_grpo 按 uid 分组、按 task_key 归一化
```

下面按阶段拆开。

### 3.1 数据集阶段：任务标签不会丢

`RLHFDataset.__getitem__()` 在构造 `input_ids`、`attention_mask` 之外，并没有删除 `problem_type`、`data_type`、`metadata` 这些原始字段。

它只会：

- 弹出 `prompt` / `answer`
- 处理 `images` 或 `videos`
- 把答案改名为 `ground_truth`

所以像 `problem_type="add"`、`problem_type="sort"`、`problem_type="temporal_seg"` 这种标签，会作为非张量字段保留在 batch 里，后续 reward 和 advantage 都能读到。

### 3.2 dataloader 阶段：按任务分桶

`verl/trainer/data_loader.py` 在 `task_homogeneous_batching=true` 时，不使用普通 `RandomSampler`，而改用 `TaskHomogeneousBatchSampler`。

这个 sampler 的逻辑是：

1. 遍历整个 HF dataset
2. 按 `task_key`，也就是 `problem_type`，把样本索引放进不同桶
3. 每个桶内部独立 shuffle
4. 按 `task_weights` 计算每个任务本 epoch 能取多少个 batch
5. 再把不同任务的 batch 交错排开

因此这个“多任务框架”的第一层任务区分，其实发生在采样器，而不是 loss。

要点：

- `task_weights` 控制的是“任务出现频率”
- 不是“同一个 batch 里每个样本的梯度权重”
- 由于 batch 同质化，单步训练基本可以视为“一个 step 只优化一个任务”

### 3.3 rollout 阶段：用 `uid` 建立 GRPO group

`RayPPOTrainer._make_batch_data()` 对每个原始样本先分配一个新的 `uid`：

```text
一个 prompt -> 一个 uid
```

随后：

1. 先对 prompt 做一次 rollout
2. 再把原 batch `repeat(n, interleave=True)`
3. 让同一个 prompt 的 `n` 个回复共享同一个 `uid`

这一步非常关键，因为后面的 GRPO/EMA-GRPO 不是按 `problem_type` 找“同一组候选”，而是按 `uid` 找“同一个 prompt 的多次采样结果”。

所以：

- `uid` 负责“GRPO group”
- `problem_type` 负责“任务类型”

这是两套不同维度的分组。

### 3.4 reward 阶段：统一入口，按任务分发

`BatchFunctionRewardManager.compute_reward()` 会把下面这些字段拼进每个 reward 输入：

- `response`
- `ground_truth`
- `data_type`
- `problem_type`

然后交给 `verl/reward_function/mixed_proxy_reward.py:compute_score`。

`mixed_proxy_reward.py` 的 `_TASK_REWARD_DISPATCH` 当前是：

- `add` -> `_choice_reward`
- `delete` -> `_choice_reward`
- `replace` -> `_choice_reward`
- `sort` -> `_sort_reward`
- `temporal_seg` -> `_temporal_seg_reward`

也就是说，这个项目现在的多任务 reward 不是“统一公式”，而是“统一入口 + 按任务分派”。

具体差异：

- `add/delete/replace`：单字母精确匹配
- `sort`：数字序列的 jigsaw displacement reward
- `temporal_seg`：复用时序分割的 F1-IoU reward

### 3.5 trainer 日志阶段：按任务单独打 reward 指标

`ray_trainer.py` 在拿到 `reward_tensor` 之后，还会根据 `batch.non_tensor_batch["problem_type"]` 重新统计每个任务的平均 reward，并记录为：

- `reward/add`
- `reward/delete`
- `reward/replace`
- `reward/sort`
- `reward/temporal_seg`

这说明当前框架对“多任务”的观测口径也是按 `problem_type` 做的。

## 4. GRPO / EMA-GRPO 在当前框架里的多任务区分逻辑

这是最关键的部分。

### 4.1 先说普通 `grpo`

普通 `compute_grpo_outcome_advantage()` 的逻辑只有一层分组：

1. 对每个样本把 token reward 求和得到 `score`
2. 按 `index` 分组，这里的 `index` 实际上传入的是 `uid`
3. 对同一组内的多个回复做：
   - 组均值
   - 组标准差
   - 标准化 `(score - group_mean) / group_std`

这里完全不看 `problem_type`。

所以普通 `grpo` 的本质是：

- 区分 prompt
- 不区分任务

如果一个 batch 恰好有多任务混在一起，普通 `grpo` 也不会为不同任务维护独立尺度。

### 4.2 当前脚本实际使用的是 `ema_grpo`

`compute_advantage()` 会把这些字段一起传给 advantage estimator：

- `token_level_rewards`
- `response_mask`
- `index = uid`
- `data_type`
- `problem_type`

这意味着 `ema_grpo` 可以同时看到：

- 哪些样本属于同一个 prompt group
- 哪些样本属于同一个任务

### 4.3 `ema_grpo` 的两层分组

`compute_ema_grpo_outcome_advantage()` 实际做了两层分组：

#### 第一层：按 `uid` 分组，做 group mean centering

对同一 prompt 的 `n` 个回复：

```text
centered_score = score - group_mean(uid)
```

这一层继承的是 GRPO 的核心思想，作用是比较“同题多个候选谁更好”。

#### 第二层：按 `task_key` 分组，做 task-level std scaling

它会把样本再按任务聚合，默认规则是：

```text
task_key = problem_type
```

只有一种特例：

```text
if problem_type == "segmentation":
    task_key = "segmentation/image" or "segmentation/video"
```

但当前脚本的数据标签是 `temporal_seg`，不是 `segmentation`，所以在这套脚本对应的数据里，真实生效的规则就是：

```text
task_key = problem_type
```

也就是：

- `add` 一套 EMA 统计
- `delete` 一套 EMA 统计
- `replace` 一套 EMA 统计
- `sort` 一套 EMA 统计
- `temporal_seg` 一套 EMA 统计

### 4.4 `ema_grpo` 的实际更新顺序

当前实现的顺序不是“先读旧统计，再更新”，而是：

1. 先把本 batch 每个样本的 token reward 求和得到 `scores`
2. 按 `uid` 做组均值中心化，得到 `centered`
3. 按任务把当前 batch 的原始 `scores` 写入 EMA 统计
4. 读取“已经包含当前 batch”的任务标准差 `task_std`
5. 用 `centered / task_std` 做缩放
6. 乘上 `response_mask` 广播回 token 维度

也就是说，当前是“先更新 EMA，再拿更新后的 std 缩放自己”。

这个实现和文件注释是一致的：它强调的是更快适应非平稳 reward 分布。

### 4.5 guard rail 保护逻辑

`ema_grpo` 还有一层保护：

- 先尝试使用任务级 `task_std`
- 如果某个 `uid` group 缩放后有值超过 `guard_abs_max`，默认 5.0
- 那么这个 group 回退到普通 GRPO 风格的 `group_std`

因此它不是完全抛弃 GRPO 的 group std，而是：

- 默认用任务级 EMA std
- 极端情况回退到组内 std

### 4.6 为什么这套逻辑适合多任务

多任务混训时，不同任务 reward 分布通常差别很大：

- 选择题 reward 基本是 0/1
- 排序题 reward 是连续值
- 时序分割 reward 也是连续值，而且分布通常更稀疏

如果直接用统一尺度，某些任务会天然方差更大，advantage 更容易主导训练。

`ema_grpo` 的处理方式是：

- 先保留 GRPO 的“同题相对比较”
- 再用“任务级历史标准差”把不同任务的 reward 尺度拉回到更接近的范围

所以它解决的是“同一 prompt 内谁更好”和“不同任务之间量纲不一致”这两个问题。

## 5. 当前架构下，多任务区分到底发生在哪几层

可以把当前系统里的“多任务区分”总结成三层：

### 第一层：采样层

通过 `TaskHomogeneousBatchSampler` 按 `problem_type` 分桶，决定每一步训练看到哪个任务。

### 第二层：reward 层

通过 `mixed_proxy_reward.compute_score()` 按 `problem_type` 分派到不同 reward 公式。

### 第三层：advantage 层

通过 `ema_grpo` 按 `uid` 做组内中心化，再按 `problem_type` 做任务级标准差归一化。

所以当前项目的多任务架构不是“共享一个完全统一的 RL 目标”，而是：

```text
共享同一个模型
+ 共享同一套 rollout / PPO 框架
+ 共享同一个 reward 入口
+ 共享同一个 actor 更新流程
+ 但在采样、reward、advantage 三处按任务分流
```

## 6. 这份脚本和当前框架之间最重要的对应关系

### 6.1 脚本里的 `task_homogeneous_batching=true`

对应训练框架中的 `TaskHomogeneousBatchSampler`。

效果：

- 每步基本只训练单一任务
- 降低多任务 reward 分布同时混入一个 batch 的不稳定性

### 6.2 脚本里的 `task_weights`

对应 sampler 中每个任务在一个 epoch 里可以取多少个 batch。

效果：

- 控制任务频率
- 不直接改 advantage，也不直接改 loss 权重

### 6.3 脚本里的 `REWARD_FUNCTION=mixed_proxy_reward`

对应 reward 层的统一 dispatch。

效果：

- 多任务共用一个 reward 接口
- 每个任务实际走不同评分规则

### 6.4 脚本里的 `ADV_ESTIMATOR=ema_grpo`

对应 advantage 层的任务级归一化。

效果：

- 同 prompt 内仍然按 GRPO group 做比较
- 不同任务各自维护 EMA 均值/标准差

## 7. 需要注意的实现细节与边界

### 7.1 `task_weights` 是近似配比，不是绝对精确配比

`TaskHomogeneousBatchSampler` 是先算每个任务最多能提供多少 batch，再用整数截断分配理想 batch 数，所以实际比例是近似值。

### 7.2 `task_homogeneous_batching` 并不是 `ema_grpo` 生效的前提

即使未来一个 batch 里混入多任务，`ema_grpo` 仍然能按任务拆分统计，因为它内部会自己按 `problem_type` 聚合。

但当前脚本开启同质 batch，会让训练更稳，也让单步日志更好解释。

### 7.3 `temporal_seg` 不会触发 `segmentation/image|video` 的特殊分支

`ema_grpo` 里写的特殊逻辑是：

```text
problem_type == "segmentation"
```

而当前脚本和混合数据使用的是：

```text
problem_type == "temporal_seg"
```

因此当前任务划分仍然完全按 `temporal_seg` 这个字符串本身处理，不会继续细分成 image/video。

在当前数据全是视频时这没有问题，但如果后面想把更多 segmentation 子类型一起混训，需要统一命名约定。

### 7.4 如果切回普通 `grpo`，任务级方差归一化会消失

那时仍然有：

- 按任务采样
- 按任务 reward

但没有：

- 按任务维护独立 advantage 尺度

这是脚本里 `ema_grpo` 最关键的价值。

## 8. 一句话总结

这份脚本本质上是在 EasyR1 上搭了一套“按 `problem_type` 分桶采样 + 按 `problem_type` 分发 reward + 按 `problem_type` 做 EMA-GRPO 标准差归一化”的多任务 RL 训练方案。

其中：

- `uid` 负责区分“同一个 prompt 的多个 rollout 候选”
- `problem_type` 负责区分“这条样本属于哪个任务”

当前框架里，GRPO 的组内比较是按 `uid` 做的，多任务尺度对齐是按 `problem_type` 做的，这两层分工就是这套多任务训练架构的核心。
