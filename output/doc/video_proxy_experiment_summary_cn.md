# VideoProxy 实验方案与阶段性观察

日期：2026-05-25

这份文档用于给接手同学快速说明：VideoProxy 这一轮实验怎么做、评测了什么、目前看到了哪些现象，以及哪些结论还需要谨慎确认。本文只汇总代码、已有报告和远端评测结果，不展开原始日志或逐样本输出。

## 1. 一句话结论

我们做的是一组多任务视频 proxy 训练实验：先分别训练 task-specific teacher，再用 OPD/MOPD 把 teacher 信号蒸到 student/目标模型上，并在外部视频理解 benchmark 上评估泛化。

当前现象比较清楚：

1. 4B CoT teacher 在内部 val 上确实能学起来，100 step 内 AOT/Seg/Logic 的 val reward 从约 `0.27-0.33` 提升到 `0.53-0.58`。
2. CoT 格式本身也学会了，尤其 Seg/Logic 到 step100 基本能稳定输出 `<thought>...</thought>`；AOT 的 repair 比例偏高，说明 128-token thought budget 对 AOT 较紧。
3. 但外部 benchmark 上，CoT 训练版本没有明显超过 direct/no-CoT/final 版本，更多是持平或小幅波动；这需要单独诊断，而不能只写成“CoT 没用”。
4. MOPD 后 QA/AoT-QA 没有出现非常稳定的跃升：Qwen3 有小幅收益，Qwen2.5 也只在 AoT-QA/Vinoground 这类指标上有局部收益，很多通用视频 QA benchmark 持平或回落。
5. 8B 不稳定地优于 4B：有些通用 benchmark 8B 更好，但在部分任务上 4B final/direct 版本持平甚至略好；不能简单认为 8B 一定更强。
6. 目前最可疑的问题不是“模型不会 CoT”或“MOPD 完全无效”，而是训练目标、teacher 路由、CoT budget、外部 QA benchmark 口径之间存在错配；这些需要后续用 ablation 拆开。

## 2. 实验分支

| 分支 | 做法 | 代码入口 | 任务组合 |
| --- | --- | --- | --- |
| Teacher no-CoT | 从已有 experiment JSONL 直接训练单任务 teacher | `video_proxy/experiments/teacher_train/qwen3_vl_4b/run_{aot,seg,logic}_nocot.sh` | AOT: `tg mcq aot`; Seg: `tg mcq hier_seg`; Logic: `tg mcq event_logic` |
| Teacher CoT | 先把 prompt/messages 转成 CoT 样式，再训练单任务 teacher | `video_proxy/experiments/teacher_train/qwen3_vl_4b/run_{aot,seg,logic}_cot.sh` | 同上，默认 `<thought>` tag |
| GRPO baseline | 不用 OPD teacher，直接跑多任务 RL baseline | `video_proxy/experiments/baselines/grpo/qwen3_vl_4b/run.sh` | full composition: `tg mcq hier_seg event_logic aot` |
| OPD | 单 teacher 或多 teacher 蒸馏 | `video_proxy/experiments/opd/*/run.sh` | 默认 recipe 是 `tg mcq aot`，MOPD preset 会覆盖 |
| MOPD 3 teachers | AOT/Seg/EventLogic 三教师，按 `problem_type` 路由 | `video_proxy/experiments/opd/*/run_mopd_3teachers.sh` | `tg mcq hier_seg event_logic aot` |
| MOPD 2 teachers | AOT/Seg 两教师 | `video_proxy/experiments/opd/*/run_mopd_2teachers.sh` | `tg mcq hier_seg aot` |

关键代码依据：

- `video_proxy/training/recipes/single_teacher_from_experiment.sh`：CoT 模式下调用 `convert_jsonl_to_cot.py` 转换 train/val；no-CoT 直接使用 source JSONL。
- `video_proxy/experiments/opd/common_mopd.sh`：定义 full composition/base R1R2 数据、teacher checkpoint 默认路径、MOPD 训练超参。
- `video_proxy/training/recipes/opd_train.sh`：OPD 使用 `TRAINING_MODE=opd`、`ROLLOUT_N=1`、`OPD_TOPK=10`、`OPD_KL_COEF=1.0`。

## 3. 训练设置

| 设置 | Teacher | CoT teacher | MOPD |
| --- | --- | --- | --- |
| 算法 | EMA-GRPO | EMA-GRPO | OPD + GRPO estimator |
| rollout | 4B/3B 默认较大，7B/8B 较小；single teacher 2GPU 默认 `N_GPU*4` | 同 teacher | 8GPU preset 默认 `ROLLOUT_BS=64`, `GLOBAL_BS=64` |
| 视频帧 | teacher recipe 默认 `MAX_FRAMES=48`，single teacher 会继承源数据 frame policy | 同 teacher | `MAX_FRAMES=256`, `MAX_PIXELS=65536` |
| response | 通用默认 `MAX_RESPONSE_LEN=256`，single teacher 可到 512 | 100-step CoT 报告使用 256 | 256 |
| KL/entropy | teacher recipe `LR=1e-6`, `KL_COEF=0.001`, `ENTROPY_COEFF=0`；single teacher 常用 `LR=5e-7`, `KL=0.01`, `entropy=0.005` | 同 single teacher | `LR=5e-7`, `KL_COEF=0.01`, `ENTROPY_COEFF=0` |
| CoT 约束 | 关闭 | `COT_BUDGET_MAX_TOKENS=128`, `<thought>...</thought>`, format reward 开 | 默认关闭 |

CoT 数据转换主要在 `video_proxy/data/scripts/convert_jsonl_to_cot.py`：MCQ 要先 reasoning 再 `<answer>`；TG 要先解释时间依据再输出 final sentence；Seg/events 会要求 shots、KEEP/MERGE/SPLIT 和 partition check。

## 4. Reward 怎么评

统一入口是 `verl/reward_function/mixed_proxy_reward.py:compute_score`。

| 任务 | problem_type | reward 口径 |
| --- | --- | --- |
| MCQ/AOT/EventLogic 选择题 | `llava_mcq`, `seg_aot_*`, `event_logic_predict_next/fill_blank` | 从 `<answer>` 提取最后一个选项字母；正确为 1，错误为 0 |
| Temporal grounding | `temporal_grounding` | 解析最终答案中的时间段；CoT 时先移除 `<think>/<thought>`；reward = temporal IoU × endpoint distance penalty |
| Hierarchical segmentation | `temporal_seg_hier_L1/L2/L3_seg` | 必须输出 `<events>`；用 F1-IoU 风格匹配 segments |
| Sort/Logic sort | `sort`, `event_logic_sort` | 必须 `<answer>`；解析数字序列，用 jigsaw displacement reward |
| CoT format | `cot_budget_debug` | missing=0，repaired/truncated=0.5，ok=1；最终 `overall = base_overall * cot_format` |

这个设计意味着 CoT 训练的最终 reward 不只是任务正确性，还会被格式预算乘法门控。如果模型推理写太长导致 repair，即使答案对，也会被打折。

## 5. 内部 CoT teacher 结果

已有报告：`output/doc/qwen3_vl_4b_cot_training_benefit_report_cn.md`。覆盖三个 100-step 4B CoT teacher：

- `qwen3_vl_4b_aot_100step_cot`
- `qwen3_vl_4b_seg_100step_cot`
- `qwen3_vl_4b_logic_100step_cot`

| Run | Val reward 0 -> 100 | 提升 | CoT format 0 -> 100 | step100 repair | 观察 |
| --- | ---: | ---: | ---: | ---: | --- |
| AOT | 0.3231 -> 0.5292 | +0.2061 | 0.7563 -> 0.9137 | 0.173 | 有收益，但 repair 高，step50 后波动 |
| Seg | 0.2708 -> 0.5452 | +0.2744 | 0.7430 -> 0.9963 | 0.007 | 提升稳定，格式几乎学会 |
| Logic | 0.3269 -> 0.5765 | +0.2496 | 0.7657 -> 0.9953 | 0.009 | 最干净，最终 val 最高 |

TG 子任务上，三个 run step100 reward 都在 `0.29-0.30` 左右，CoT 格式质量很好，但时间边界仍会漂移。这说明 CoT 能解释“为什么是这段”，但不等于精确定位能力已经解决。

## 6. 外部评测覆盖

远端结果主要在：

- `/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct_qwen25_fixed`
- `/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct_qwen3_cot`
- `/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct_final_union`
- `/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct_final_1`

评测 benchmark 包括：

`Video-MME`, `MLVU_MCQ`, `MVBench`, `AoTBench_QA`, `AoTBench_ReverseFilm`, `AoTBench_UCF101`, `AoTBench_Rtime_t2v/v2t`, `TimeLensBench`, `LongVideoBench`, `TempCompass_MCQ`, `Video_Holmes`, `Video-TT`, `Vinoground` 等。

下面只列目前聚合脚本可稳定解析的百分制结果；空白代表该目录没有可稳定解析的 summary，或该 benchmark 文件采用了不同尺度。

### 6.1 Qwen3 final 主表

| Model | Video-MME | MLVU | MVBench | AoT-QA | Rtime t2v | Rtime v2t | LongVideo | TempCompass | Holmes | Vinoground |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B Instruct | 65.0 | 56.8 | 62.5 | 49.3 | 53.5 | 60.4 | 60.6 | 75.0 | 40.0 | 60.8 |
| Qwen3-8B Instruct | 66.7 | 58.0 | 64.1 | 52.2 | 56.2 | 61.6 | 59.9 | 76.7 | 40.0 | 60.3 |
| Qwen3-4B MOPD step583 | 65.3 | 58.4 | 63.9 | 51.6 | 51.6 | 61.8 | 61.6 | 75.1 | 40.3 | 63.2 |
| Qwen3-8B MOPD step583 | 67.5 | 56.8 | 65.4 | 53.7 | 51.4 | 61.8 | 60.6 | 77.2 | 40.1 | 61.3 |
| Qwen3-4B R2 | 65.4 | 59.8 | 63.8 | 52.9 | 51.7 | 62.4 | 61.8 | 75.3 | 40.6 | 63.3 |
| VideoSSR-8B | 68.1 | 58.6 | 66.5 | 57.0 | 55.3 | 62.0 | 60.9 | 76.7 | 40.2 | 63.5 |

观察：

- 8B base 在 Video-MME/MVBench/AoT-QA/Rtime 上通常比 4B base 高，但不是所有任务都高，例如 LongVideo/Vinoground 上 4B MOPD/R2 更好。
- 4B MOPD step583 相比 4B base 在 MLVU、MVBench、AoT-QA、LongVideo、Vinoground 有收益，但 Video-MME 几乎持平。
- 8B MOPD step583 相比 8B base Video-MME/MVBench/AoT-QA/TempCompass 有收益，但 MLVU、Rtime t2v、Vinoground 不稳定。
- 所以“8B 有的还没有 4B 好”是成立的，但需要按 benchmark 说，不能全局下结论。

### 6.2 Qwen3 CoT 与 final union

| Model | Video-MME | MVBench | Vinoground |
| --- | ---: | ---: | ---: |
| Qwen3-4B SegCoT-V2 | 65.4 | 63.2 | 61.7 |
| Qwen3-4B AoTCoT | 65.5 | 63.3 | 61.0 |
| Qwen3-4B LogCoT | 65.3 | 63.2 | 61.5 |
| Qwen3-4B SegCoT-V1 | 65.4 | 63.4 | 61.5 |
| Qwen3-4B R2-CoT final union | 65.5 |  | 62.6 |

对比 Qwen3-4B R2/direct/final：

- CoT 版本 Video-MME 大约 `65.3-65.5`，与 4B R2 `65.4` 和 4B MOPD `65.3` 基本持平。
- CoT 版本 MVBench 大约 `63.2-63.4`，低于 4B MOPD step583 `63.9` 和 4B R2 `63.8`。
- CoT 版本 Vinoground `61.0-61.7`，低于 4B MOPD/R2 的 `63.2-63.3`；final union R2-CoT 到 `62.6`，仍未明显超过 direct/final。

这支持一个谨慎判断：CoT teacher 在内部 reward 上能学到格式和部分能力，但外部 benchmark 的泛化收益不如直接训练/最终 MOPD 明显。

### 6.3 Qwen2.5 fixed

| Model | Video-MME | MLVU | MVBench | AoT-QA | Rtime t2v | Rtime v2t | LongVideo | TempCompass | Holmes | Vinoground |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B Instruct | 62.8 | 56.4 | 63.4 | 38.9 | 55.1 | 58.8 | 59.2 | 74.9 | 23.0 | 57.4 |
| Qwen2.5-7B MOPD | 62.1 | 55.8 | 63.6 | 40.7 | 52.2 | 57.8 | 58.8 | 74.6 | 22.8 | 60.5 |
| Qwen2.5-7B base-nextvideo | 62.7 |  | 64.1 |  |  |  |  |  |  | 59.3 |
| Qwen2.5-7B base-llavavideo | 62.9 |  | 63.8 |  |  |  |  |  |  | 59.7 |
| VidBridge-R1 | 63.3 |  | 63.7 |  |  |  |  |  |  | 58.4 |

观察：

- Qwen2.5 MOPD 对 AoT-QA 和 Vinoground 有收益，但 Video-MME/MLVU/Rtime/LongVideo/TempCompass/Holmes 没有明显收益。
- 这说明蒸馏/训练并非无条件提升，某些 benchmark 上会损失通用能力或出现 trade-off。

### 6.4 QA/MOPD 增益为什么看起来不明显

如果只看 AoT-QA，MOPD 并不是完全没涨，但涨幅不够稳定，也没有扩散到大多数 QA benchmark：

| 对比 | AoT-QA delta | 同时出现的问题 |
| --- | ---: | --- |
| Qwen3-4B MOPD step583 vs 4B Instruct | `+2.3` | Video-MME 几乎持平，Rtime t2v 下降 |
| Qwen3-8B MOPD step583 vs 8B Instruct | `+1.5` | MLVU 下降，Rtime t2v 下降，Vinoground 不如 4B |
| Qwen3-4B R2 vs 4B Instruct | `+3.6` | 主要是阶段性 R2 更好，不能直接归因于 MOPD 机制本身 |
| Qwen2.5-7B MOPD vs 7B Instruct | `+1.8` | Video-MME/MLVU/Rtime/LongVideo/TempCompass/Holmes 回落 |

因此更准确的说法是：MOPD 对与 proxy task 更接近的 AoT-QA 有局部正收益，但对一般视频 QA 能力没有形成稳定提升，甚至可能牺牲部分通用能力。

这件事需要诊断，因为 MOPD 训练代码本身更像“模仿 teacher 的 token 分布”，不是直接最大化外部 benchmark 的 QA accuracy：

- OPD/MOPD 训练时 `ROLLOUT_N=1`，不是 GRPO 那种多 sample 相对优势学习；见 `video_proxy/training/recipes/opd_train.sh:17-23` 和 `video_proxy/experiments/opd/common_mopd.sh:107-118`。
- actor 更新只对 teacher top-k token 分布做 KL 蒸馏：`teacher_probs * (teacher_logp - student_logp)`，再乘 `OPD_KL_COEF`；见 `verl/workers/actor/dp_actor.py:472-484`。
- 多 teacher 路由按 `problem_type` 粗分：`aot`/`seg`/`eventlogic`。其中 `llava_mcq`、`temporal_grounding` 也会被路由到 `aot` teacher；见 `verl/workers/teacher_routing.py:51-55`。如果外部 QA 的能力需求不等同于 AOT proxy，这个路由会带来错配。
- MOPD 默认 teacher 来自 task-specific no-CoT checkpoint，路径是 `composition_base_aot_aot10k_mf256_ema`、`composition_base_seg_hier10k_mf256_ema`、`composition_base_logic_el10k_mf256_ema`；见 `video_proxy/experiments/opd/common_mopd.sh:130-140`。这解释了为什么它更可能提升 AOT/Seg/Logic 相近任务，而不是所有视频 QA。
- EMA-GRPO 的任务分桶把 `llava_mcq`、AoT、event logic predict/fill 都归为 `mcq`；见 `verl/trainer/core_algos.py:334-356`。如果 QA benchmark 的错误类型和 proxy MCQ 不一致，训练 reward 提升不一定能变成外部 QA 提升。

当前应优先排查三件事：

1. 按 benchmark 拆分：把 AoT-QA、Video-MME、MLVU、MVBench、LongVideoBench 分开看，不要用一个平均数判断 MOPD。
2. 按 `problem_type` 统计 MOPD 训练 reward、teacher route、teacher top-k prob mass，确认 QA/MCQ 样本是否被 AOT teacher 过度主导。
3. 做 same-step ablation：同一 base、同一数据、同一 checkpoint step，对比 GRPO direct、single AOT teacher OPD、MOPD 2-teacher、MOPD 3-teacher。

## 7. CoT 为什么可能没有 direct 好

目前更像是训练目标/预算/评测错配，而不是简单的模型容量问题。

可能原因：

1. **CoT 格式门控会放大格式问题。** `overall = base_overall * cot_format`，AOT step100 repair 仍有 `17.3%`。这些样本即便答案可解析，也会被截断/repair 打折。
2. **AOT 任务天然更耗 reasoning budget。** AOT 需要比较 clip 和文本/动作顺序，128 token 容易不够；报告中 AOT thought 平均 68.2 words，高于 Seg/Logic。
3. **内部 reward 和外部 benchmark 不完全一致。** 内部 val 能证明模型会做 proxy task 和写 CoT，但外部 Video-MME/MVBench/Vinoground 更看泛化能力，CoT 文本不一定转化为更好的最终答案。
4. **TG 的瓶颈是边界精度。** CoT 能给视觉证据，但 final start/end 仍漂移；TG step50 后基本平台。
5. **MOPD/teacher 路由可能更适合 no-CoT teacher。** 现有 MOPD 默认 teacher path 多来自 no-CoT 或 direct teacher；如果 student 训练/评测最终只看短答案，长 CoT teacher 可能带来分布差异。

更具体的诊断点：

| 现象 | 代码/结果依据 | 可疑原因 | 下一步验证 |
| --- | --- | --- | --- |
| 内部 CoT val reward 明显涨，但外部 benchmark 没明显涨 | AOT/Seg/Logic val reward 到 `0.53-0.58`；Video-MME/MVBench/Vinoground 基本持平或低于 direct | proxy task 学会了，但外部 QA 的评测只看 final answer，CoT 文本不一定带来泛化 | 同 checkpoint 用 no-CoT prompt 和 CoT prompt 分别评测外部 benchmark |
| AOT CoT repair 比例高 | AOT step100 repair `0.173`，Seg/Logic 约 `0.007/0.009` | AOT reasoning 更长，128-token thought budget 不够；repair 会把 reward 乘到 0.5 | AOT 单独跑 128/192/256 budget ablation，看 repair 与外部 AoT-QA/MVBench 是否同步改善 |
| CoT 错误会被乘法放大 | `_apply_cot_format_reward` 中 `overall = base_overall * cot_format` | 格式错误和答案错误绑定在一起，训练可能过度优化短格式而不是任务判断 | 记录 `overall_base` 与 `overall` 的 gap，统计被 format gate 打折但答案正确的样本比例 |
| TG CoT 看起来会解释，但边界仍漂 | TG step100 reward 约 `0.29-0.30` | CoT 证据描述无法直接约束 start/end 边界 | 单独评估 endpoint error、IoU 分布，而不是只看平均 reward |
| final union 的 R2-CoT 仍未超过 direct/final | R2-CoT Video-MME `65.5`，Vinoground `62.6`；4B R2/direct Vinoground `63.3` | union 可能混入多种 prompt/teacher 分布，未形成稳定收益 | 对 union 拆来源，分别评估 SegCoT/AoTCoT/LogicCoT 对应任务集 |

## 8. 建议下一步

1. 做严格 ablation：同模型、同数据、同 step、同 decoding，对比 no-CoT teacher vs CoT teacher。
2. 对 AOT CoT 单独放宽 budget 或缩短 CoT 指令，比较 repair ratio 是否下降。
3. 对 TG 加强 final answer 约束，或引入更直接的 boundary supervision，而不是只依赖 CoT evidence。
4. 评测表分两套口径：内部 val reward 说明 proxy 学习；外部 benchmark 说明泛化，避免混在一起下结论。
5. 对 8B 与 4B 的结论按 benchmark 拆开：列出 8B 胜、4B 胜、持平三类，避免“模型越大越好”的笼统表述。
6. 对 QA/MOPD 单独做诊断表：按 `problem_type`、teacher route、benchmark family、checkpoint step 拆 delta，重点看 AoT-QA 的局部收益有没有牺牲 Video-MME/MLVU/LongVideo。

## 9. 证据路径

本地代码：

- `/Users/lostgreen/Desktop/Codes/VideoProxy/train/video_proxy/experiments/teacher_train/README.md`
- `/Users/lostgreen/Desktop/Codes/VideoProxy/train/video_proxy/training/recipes/single_teacher_from_experiment.sh`
- `/Users/lostgreen/Desktop/Codes/VideoProxy/train/video_proxy/experiments/opd/common_mopd.sh`
- `/Users/lostgreen/Desktop/Codes/VideoProxy/train/video_proxy/training/recipes/opd_train.sh`
- `/Users/lostgreen/Desktop/Codes/VideoProxy/train/verl/reward_function/mixed_proxy_reward.py`
- `/Users/lostgreen/Desktop/Codes/VideoProxy/train/verl/reward_function/temporal_grounding_reward.py`
- `/Users/lostgreen/Desktop/Codes/VideoProxy/train/output/doc/qwen3_vl_4b_cot_training_benefit_report_cn.md`

远端结果：

- `/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct_final_1`
- `/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct_qwen3_cot`
- `/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct_final_union`
- `/m2v_intern/xuboshen/zgw/VideoProxyMixed/eval_direct_qwen25_fixed`
- `/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_{aot,seg,logic}_100step_cot`
