# Multi-Teacher OPD 实现说明

本文说明当前分支如何把 single-teacher OPD 扩展成 multi-teacher OPD，以及 OPD 的训练原理和本次对 verl 框架做的改动。

参考资料：

- verl 官方 OPD 文档：<https://github.com/verl-project/verl/blob/main/docs/advance/async-on-policy-distill.md>
- verl Multi-Teacher OPD PR：<https://github.com/verl-project/verl/pull/6051>

## 1. OPD 是什么

OPD 是 On-Policy Distillation，即“在线策略蒸馏”。它的核心思路是：

1. student policy 先基于当前参数生成回复；
2. teacher 不生成完整答案，而是对 student 已经生成的 token 序列逐位置打分；
3. teacher 在每个 response token 预测位置返回 top-k token 分布；
4. student 在同一批 on-policy token 上拟合 teacher 的 top-k 分布。

和普通 SFT 的区别：

- SFT 学的是离线 teacher answer；
- OPD 学的是 student 自己采样轨迹上的 teacher next-token distribution；
- 所以 OPD 更接近 student 当前状态分布，distribution mismatch 更小。

和 RL/GRPO 的区别：

- OPD 不需要 reward；
- 不需要 advantage；
- 不需要 critic；
- 一般 `rollout.n=1`；
- loss 直接来自 teacher-student sparse KL。

本分支中的 OPD loss 是 sparse forward KL：

```text
L_opd = mean_i mask_i * sum_j p_T(j | prefix_i) * (log p_T(j | prefix_i) - log p_S(j | prefix_i))
```

其中：

- `i` 是 response token 位置；
- `j` 是 teacher top-k token；
- `p_T` 是 teacher top-k 概率；
- `p_S` 是 student 在 teacher top-k token id 上 gather 出来的概率；
- `mask_i` 排除 padding token；
- 最终 loss 再乘 `algorithm.opd_kl_coef`。

## 2. verl 官方 multi-teacher OPD 的设计

verl 上游 PR 的 multi-teacher OPD 设计更偏生产级：

- 使用 `distillation.teacher_models` 配置多个 teacher；
- 使用 `distillation.teacher_key` 从样本字段中选择 teacher；
- 每个 teacher 有独立 rollout/server replica；
- 通过 `MultiTeacherModelManager` 管理 teacher resource pool；
- teacher pool 和 student/rollout pool 是分开的。

这种设计适合多机多卡或至少有额外 teacher GPU 的场景。问题是当前目标是“2 卡能跑”，如果直接采用上游独立 teacher pool，teacher 会额外占 GPU，和 student/rollout 抢显存，基本不满足当前约束。

因此本分支没有完整移植上游独立 teacher server/pool，而是保留 single-teacher OPD 的 FSDP ref 路径，在同一个 `ActorRolloutRef` worker 内挂多个 ref teacher，并通过 CPU offload 控制显存。

## 3. 当前分支的实现取舍

目标：

- 2 卡 smoke run 可运行；
- AoT 样本使用 AoT teacher；
- segmentation 样本使用 segmentation teacher；
- event logic 样本使用 eventlogic teacher；
- 不使用的 teacher offload 到 CPU，避免多个 teacher 同时占 GPU。

核心取舍：

- 不新增独立 Ray teacher resource pool；
- 不新增 async teacher server；
- 仍使用现有 `compute_ref_topk_log_probs()` 链路；
- 每个 teacher 是一个 FSDP ref module；
- 同一步 batch 内如果出现多个任务，就按 teacher 分成多个 sub-batch，依次 load / forward / offload；
- 这样牺牲吞吐，但换取 2 卡可跑和实现风险更低。

运行时大致流程：

```text
student rollout(n=1)
  -> batch balance
  -> compute_ref_topk_log_probs
       -> read non_tensor_batch[problem_type]
       -> group sample indices by teacher
       -> load active teacher FSDP params to GPU
       -> compute teacher_topk_logps / teacher_topk_indices
       -> reshard + offload active teacher to CPU
       -> scatter sub-batch outputs back to original batch order
  -> actor update with OPD sparse KL
```

## 4. 路由规则

默认按 `problem_type` 路由。

当前默认规则在 `verl/workers/teacher_routing.py`：

| `problem_type` | teacher |
|---|---|
| 包含 `aot`，例如 `seg_aot_action_v2t_3way` | `aot` |
| `temporal_grounding` / `tg` / `grounding` | `aot` |
| `llava_mcq` | `aot` |
| 以 `temporal_seg` 或 `hier_seg` 开头 | `seg` |
| 以 `event_logic` 开头 | `eventlogic` |

如果任务名不符合默认规则，可以通过 `worker.ref.teacher_task_map` 做精确或前缀覆盖。前缀规则使用 `*` 结尾，例如：

```text
seg_aot_* -> aot
event_logic_* -> eventlogic
```

如果没有匹配规则且配置了多个 teacher，代码会 fail closed，直接报错，不会静默落到错误 teacher。

## 5. 框架改动

### 5.1 `RefConfig` 支持多个 teacher

文件：`verl/workers/actor/config.py`

新增字段：

```python
teacher_models: Dict[str, ModelConfig]
teacher_key: str = "problem_type"
teacher_task_map: Dict[str, str]
default_teacher: Optional[str]
```

含义：

- `teacher_models`：teacher 名称到模型路径配置的映射；
- `teacher_key`：从 `DataProto.non_tensor_batch` 读取哪个字段做路由；
- `teacher_task_map`：可选任务到 teacher 的覆盖表；
- `default_teacher`：可选兜底 teacher。

### 5.2 `WorkerConfig.post_init()` 保留 single-teacher 兼容

文件：`verl/workers/config.py`

原逻辑是：

```python
if self.ref.model.model_path is None:
    self.ref.model = deepcopy(self.actor.model)
```

这对 single-teacher KL/OPD 是合理的，因为没有显式 ref model 时默认用 actor model。

multi-teacher 下不能这么做，否则 `worker.ref.model` 会被 actor model 填上，并和 `teacher_models` 语义冲突。

现在改成：

```python
if self.ref.model.model_path is None and not self.ref.teacher_models:
    self.ref.model = deepcopy(self.actor.model)
```

同时会 normalize `teacher_models` 中的 `ModelConfig`，保证 tokenizer/path post-init 行为一致。

### 5.3 新增 teacher 路由 helper

文件：`verl/workers/teacher_routing.py`

核心函数：

```python
resolve_opd_teacher_name(
    routing_value,
    teacher_names,
    task_map=None,
    default_teacher=None,
)
```

它只负责“任务名 -> teacher 名”的纯逻辑，不依赖 Ray/FSDP/Torch，便于 CPU 单测。

### 5.4 `FSDPWorker` 支持多个 ref module

文件：`verl/workers/fsdp_workers.py`

新增状态：

```python
self.ref_fsdp_modules = {}
self.ref_policies = {}
```

初始化时，如果配置了 `worker.ref.teacher_models`：

```python
for teacher_name, teacher_model_config in self.config.ref.teacher_models.items():
    self._build_model_optimizer(..., role="ref", ref_name=teacher_name)
```

每个 teacher 都会构建自己的 FSDP module 和 `DataParallelPPOActor` wrapper。

### 5.5 `compute_ref_topk_log_probs()` 增加 multi-teacher 分支

原 single-teacher 流程：

```text
load ref
compute top-k
offload ref
```

新流程：

```python
if self.ref_fsdp_modules:
    output = self._compute_multi_teacher_ref_topk_log_probs(data)
else:
    output = self._compute_ref_topk_log_probs_with_module(data, self.ref_fsdp_module, self.ref_policy)
```

multi-teacher 分支会：

1. 从 `data.non_tensor_batch[worker.ref.teacher_key]` 读取任务标签；
2. 调用 `resolve_opd_teacher_name()` 得到 teacher 名；
3. 按 teacher 聚合样本 index；
4. 对每个 teacher 的 sub-batch 调用 `_compute_ref_topk_log_probs_with_module()`；
5. 把输出 scatter 回 full batch tensor。

每个 teacher forward 外层都有 `try/finally`：

```python
if self._use_ref_param_offload:
    load_fsdp_model(ref_module)
try:
    ...
finally:
    if self._use_ref_param_offload:
        offload_fsdp_model(ref_module)
```

这样即使 teacher forward 报错，也会尽量把参数 offload 回 CPU。

## 6. 训练脚本改动

### 6.1 新增 2 卡 launcher

文件：`local_scripts/run_multi_teacher_opd.sh`

可覆盖默认 teacher：

```bash
AOT_TEACHER_MODEL_PATH=/path/to/aot_teacher
SEG_TEACHER_MODEL_PATH=/path/to/seg_teacher
EVENTLOGIC_TEACHER_MODEL_PATH=/path/to/eventlogic_teacher
```

当前 2 卡 launcher 已内置默认 teacher 路径：

```bash
SEG_TEACHER_MODEL_PATH=/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations/composition_base_seg_hier10k_mf256_ema/global_step_250/actor/huggingface
AOT_TEACHER_MODEL_PATH=/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations/composition_base_aot_aot10k_mf256_ema/global_step_200/actor/huggingface
EVENTLOGIC_TEACHER_MODEL_PATH=/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations/composition_base_aot_logic_aot10k_el10k_mf256_ema/global_step_300/actor/huggingface
```

默认关键参数：

```bash
N_GPUS_PER_NODE=2
TP_SIZE=1
ROLLOUT_BS=16
GLOBAL_BS=16
ROLLOUT_N=1
ROLLOUT_TEMPERATURE=1.0
TRAINING_MODE=opd
DISABLE_KL=false
USE_KL_LOSS=false
ONLINE_FILTERING=false
OPD_TOPK=10
OPD_KL_COEF=1.0
MAX_FRAMES=256
MAX_PIXELS=65536
MAX_STEPS=50
SAVE_FREQ=50
SAVE_LIMIT=3
ACTOR_OFFLOAD_PARAMS=true
ACTOR_OFFLOAD_OPTIMIZER=true
REF_OFFLOAD_PARAMS=true
```

默认训练数据直接使用 task-composition 产出的全组合 `mf256` 数据：

```bash
TRAIN_FILE=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/experiments/composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema/train.jsonl
TEST_FILE=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/multi_task/experiments/composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema/val.jsonl
TASKS="tg mcq hier_seg event_logic aot"
```

### 6.2 `run_multi_task.sh` 支持 multi-teacher 参数

文件：`local_scripts/run_multi_task.sh`

新增 CLI 参数生成：

```bash
+worker.ref.teacher_models.aot.model_path="${AOT_TEACHER_MODEL_PATH}"
+worker.ref.teacher_models.seg.model_path="${SEG_TEACHER_MODEL_PATH}"
+worker.ref.teacher_models.eventlogic.model_path="${EVENTLOGIC_TEACHER_MODEL_PATH}"
worker.ref.teacher_key="${OPD_TEACHER_KEY}"
worker.ref.offload.offload_params="${REF_OFFLOAD_PARAMS}"
```

同时保持 single-teacher 参数兼容：

```bash
worker.ref.model.model_path="${TEACHER_MODEL_PATH}"
```

脚本会禁止同时设置 `TEACHER_MODEL_PATH` 和 multi-teacher model path，避免配置含义不清。

## 7. 运行方式

最小 2 卡 smoke run：

```bash
bash local_scripts/run_multi_teacher_opd.sh
```

常用可调参数：

```bash
MODEL_PATH=/path/to/student
EXP_NAME=multi_teacher_opd_2gpu_mf256_sanity
ROLLOUT_BS=16
GLOBAL_BS=16
MB_PER_UPDATE=1
MB_PER_EXP=1
OPD_TOPK=10
OPD_KL_COEF=1.0
ROLLOUT_GPU_MEM_UTIL=0.35
```

如果 teacher tokenizer path 和 model path 不同，也可以设置：

```bash
AOT_TEACHER_TOKENIZER_PATH=/path/to/aot_tokenizer
SEG_TEACHER_TOKENIZER_PATH=/path/to/seg_tokenizer
EVENTLOGIC_TEACHER_TOKENIZER_PATH=/path/to/eventlogic_tokenizer
```

但当前 FSDP ref 路径的 `input_ids` 来自 student tokenizer。实际训练时应确保 student 和 teachers 使用兼容 tokenizer/vocab，最好来自同一模型家族和同一 tokenizer。否则 teacher 看到的 token id 语义可能不一致。

## 8. 为什么能降低显存

关键配置：

```bash
REF_OFFLOAD_PARAMS=true
ACTOR_OFFLOAD_PARAMS=true
ACTOR_OFFLOAD_OPTIMIZER=true
```

启动时每个 teacher 都会构建 FSDP module，但 ref params 会 offload 到 CPU。

teacher 计算时：

```text
load active teacher -> forward top-k -> reshard -> offload active teacher
```

因此 GPU 上不会同时常驻 AoT / seg / eventlogic 三个 teacher。代价是：

- CPU 内存占用更高；
- teacher 切换有 load/offload 开销；
- 如果一个 batch 同时包含多个任务，一个 step 内会做多次 teacher forward。

这是为了 2 卡可跑做出的明确取舍。

## 9. 当前限制

1. **不是上游完整 async multi-teacher 架构。**  
   当前实现没有独立 teacher pool，也没有 teacher server replica。吞吐低于上游 PR 的设计，但更适合 2 卡受限环境。

2. **teacher 共享 student token ids。**  
   当前 `DataProto.input_ids` 是 student tokenizer 的结果。teacher 模型必须能正确解释这些 token ids。

3. **CPU 内存压力会变大。**  
   三个 teacher 都会初始化为 FSDP module，只是参数 offload 到 CPU。需要机器 CPU RAM 足够。

4. **batch 内多任务会串行 teacher forward。**  
   如果想减少 teacher 切换，可以考虑打开 task-homogeneous batching，让单个 step 尽量只含一种任务。

5. **OPD 不走 reward/advantage。**  
   当前 OPD 分支跳过 reward、critic、advantage 和 online filtering。相关指标要看 `actor/opd_loss`、`opd/teacher_topk_prob_mass/*` 等 OPD 指标。

## 10. 验证覆盖

新增/更新测试：

- `tests/test_opd_teacher_routing.py`
- `tests/test_single_teacher_opd_config.py`

已验证：

```bash
PYTHONPATH=. pytest tests/test_opd_teacher_routing.py tests/test_single_teacher_opd_config.py -q
```

结果：

```text
11 passed
```

同时验证：

```bash
python -m ruff check tests/test_opd_teacher_routing.py tests/test_single_teacher_opd_config.py \
  verl/workers/teacher_routing.py verl/workers/actor/config.py verl/workers/config.py verl/workers/fsdp_workers.py

bash -n local_scripts/run_multi_teacher_opd.sh local_scripts/run_multi_task.sh local_scripts/multi_task_common.sh
git diff --check
```

均通过。
