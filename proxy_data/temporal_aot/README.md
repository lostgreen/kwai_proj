# Temporal AoT Scripts

这个目录现在默认复用已经切好的 YouCook2 event clips，而不是再从 `proxy_train_easyr1.jsonl` 里反查视频。

推荐输入是现成的 `extracted_clips_database.json`，格式类似：

```json
{
  "GLd3aX16zBg": [
    {
      "clip_path": "/m2v_intern/xuboshen/zgw/data/youcook2_event_clips/training/113/GLd3aX16zBg/GLd3aX16zBg_event00_90_102.mp4",
      "original_video_id": "GLd3aX16zBg",
      "recipe_type": "113",
      "subset": "training",
      "sentence": "spread margarine on two slices of white bread",
      "segment_in_original": [90, 102],
      "event_id": 0,
      "sequence_index": 0
    }
  ]
}
```

核心假设：

1. 以单个 event clip 为基本单位
2. forward clip 直接复用现成切好的 `clip_path`
3. `reverse` 指该 event clip 的帧级倒放
4. `shuffle` 指该 event clip 按固定时长（默认 2s）切段后随机重排
5. 如需 `T2V`，输入视频仍然是 `forward + black gap + reverse`

---

## 整体流程

按顺序依次运行以下脚本：

**Step 1 — 生成 manifest + reverse/shuffle/composite 视频素材**

```bash
python proxy_data/temporal_aot/build_event_aot_data.py \
  --clip-db-json /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/extracted_clips_database.json \
  --output-jsonl /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/aot_event_manifest.jsonl \
  --reverse-dir /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/reverse_clips \
  --shuffle-dir /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/shuffle_clips \
  --composite-dir /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/composite_clips \
  --make-reverse \
  --make-composite \
  --make-shuffle \
  --shuffle-segment-sec 2.0 \
  --min-shuffle-segments 3 \
  --max-samples 100 \
  --build-workers 8 \
  --invalid-report-jsonl proxy_data/temporal_aot/data/aot_invalid_clips.jsonl \
  --bad-samples-jsonl proxy_data/temporal_aot/data/aot_invalid_clips.jsonl \
  --subset training
```

说明：
- `--build-workers 8`：并行 ffmpeg，可大幅缩短生成时间
- 所有 ffmpeg 输出统一使用 `-preset ultrafast -crf 23 -threads 1`，单 clip 编码速度约为默认 medium 的 5-8 倍
- 每个 ffmpeg 输出完成后自动做 `ffprobe` 可读性验证；如果 ultrafast 产出不可读，自动回退到 `-preset medium` 重编码
- `--make-shuffle`：对 duration ≥ `min-shuffle-segments × shuffle-segment-sec`（默认 6s）的 clip 生成对应的时间乱序视频，保存为 `{clip_key}_shuf.mp4`
- 不满足最小分段数的 clip 会跳过 shuffle 生成，仍正常写入 manifest（`shuffle_video_path` 为空字符串）

**Step 2 — VLM 标注 caption + direction_clear**

```bash
python proxy_data/temporal_aot/annotate_event_captions.py \
  --manifest-jsonl proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --output-dir proxy_data/temporal_aot/data/aot_annotations \
  --api-base http://localhost:8000/v1 \
  --model Qwen3-VL-7B \
  --workers 8 \
  --fwd-rev-fps 1.0 \
  --shuffle-fps 2.0
```

VLM 会对每个 clip 输出 `caption`、`confidence`、`direction_clear` 三个字段。
`direction_clear=false` 表示该动作（如搅拌、揉面）在倒放后外观几乎相同，VLM 自动标记为方向不明显，后续会被过滤。

帧采样策略：
- Forward / Reverse：**1 fps** 采样（`--fwd-rev-fps 1.0`），10s 视频 → 10 帧，足够描述整体方向
- Shuffle：**2 fps** 采样（`--shuffle-fps 2.0`），与训练时的输入帧率对齐，保证 VLM 看到的内容和模型训练时的输入一致
- Shuffle 帧采样额外做 **segment-aware 均衡**：保证每个 2s 段至少被采到 1 帧，避免长视频某些段完全没有帧覆盖
- `--max-frames` 作为硬上限（默认 32），防止超长视频产生过多帧

Shuffle 标注 prompt 设计：
- 告知 VLM **段数**和**每段时长**（如 "5 segments of approximately 2 seconds each"），引导按段逐段描述
- 要求对没有明显变化的段显式标注（如 "the mixture remains the same"），增强与 forward caption 的区分度
- 关注具体状态变化（颜色、形状、位置、数量），而非泛泛的活动标签

**Step 3 — 构造 V2T / T2V / 4-way MCQ 训练数据**

```bash
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --caption-pairs proxy_data/temporal_aot/data/aot_annotations/caption_pairs.jsonl \
  --v2t-output proxy_data/temporal_aot/data/aot_annotations/v2t_train.jsonl \
  --t2v-output proxy_data/temporal_aot/data/aot_annotations/t2v_train.jsonl \
  --fourway-output proxy_data/temporal_aot/data/aot_annotations/4way_train.jsonl
```

过滤逻辑（按序）：
1. `forward_confidence` 或 `reverse_confidence` 低于 `--min-confidence`（默认 0.6）→ 丢弃
2. `forward_caption == reverse_caption`（正反描述无差异）→ 丢弃
3. `forward_direction_clear=False` 或 `reverse_direction_clear=False`（VLM 判断方向不明显）→ 丢弃（可用 `--no-require-direction-clear` 关闭）

4-way MCQ (`aot_4way_v2t`) 结构：
- 展示一个视频（forward 或 shuffle），提供 4 个 caption 选项
- A/B/C/D 分别对应 forward / reverse / shuffle / hard-negative-caption（随机顺序）
- hard-negative 来自同一 `recipe_type` 的其他 clip 的 forward caption
- 只有当 caption_pairs 中包含 `shuffle_caption` 且 `--fourway-output` 已指定时才生成

**Step 4 — 混合 YouCook2 时序分割数据**

```bash
python proxy_data/temporal_aot/mix_aot_with_youcook2.py \
  --seg-jsonl proxy_data/youcook2_train_easyr1.jsonl \
  --v2t-jsonl proxy_data/temporal_aot/data/aot_annotations/v2t_train.jsonl \
  --t2v-jsonl proxy_data/temporal_aot/data/aot_annotations/t2v_train.jsonl \
  --train-output proxy_data/temporal_aot/data/mixed_aot_train.jsonl \
  --val-output proxy_data/temporal_aot/data/mixed_aot_val.jsonl
```

**Step 5 — （可选）离线过滤后做 A/B 答案重平衡**

```bash
python proxy_data/temporal_aot/rebalance_aot_answers.py \
  --input-jsonl proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.jsonl \
  --output-jsonl proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.balanced.jsonl
```

---

## 脚本说明

| 脚本 | 职责 |
|------|------|
| `build_event_aot_data.py` | 读取 clip DB，按 `clip_path` 去重，生成 manifest；按需生成 reverse / shuffle / composite 视频 |
| `annotate_event_captions.py` | 读取 manifest，对 forward（1fps）/ reverse（1fps）/ shuffle（2fps segment-aware）视频抽帧并调用 VLM 标注 caption |
| `build_aot_mcq.py` | 读取 manifest + caption pairs，过滤后构造 `aot_v2t` / `aot_t2v` / `aot_4way_v2t` 训练数据 |
| `mix_aot_with_youcook2.py` | 把 V2T / T2V 与 YouCook2 时序分割数据合并为统一 train/val split |
| `prompts.py` | 统一维护标注和 MCQ 用的 prompt 模板（含 `direction_clear` 判断 + shuffle 段感知描述） |
| `rebalance_aot_answers.py` | 在 offline filter 之后对保留的 AOT 样本做 A/B 选项重排，把最终答案分布尽量打平，不丢样本 |

---

## 脚本逻辑

### 1. `build_event_aot_data.py`

职责：从现成 event clip 标注构造 AoT manifest，并补充 reverse/composite/shuffle 素材。

推荐输入：

- `--clip-db-json /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/extracted_clips_database.json`

兼容输入：

- `--input-jsonl proxy_data/proxy_train_easyr1.jsonl`

实际逻辑：

1. 二选一读取输入源（`--clip-db-json` 或 `--input-jsonl`）
2. 合并已有标注字段（`original_video_id`, `recipe_type`, `subset`, `sentence`, `segment_in_original` 等）
3. 过滤条件：
   - 路径去重（同一个 `clip_path` 只保留一次）
   - `--bad-samples-jsonl` 跳过已知坏样本
   - `duration_sec >= --min-duration`
   - `ffprobe` 检查视频可读 + 有 video stream
   - 标注时长与实际时长差值 ≤ `--max-duration-diff-sec`
4. 打乱顺序后按 `--max-samples` 截断
5. 按需生成素材：
   - `--make-reverse`：`ffmpeg -vf reverse -an`，保存到 `--reverse-dir/{clip_key}_rev.mp4`
   - `--make-shuffle`：按 `--shuffle-segment-sec`（默认 2s）切段后随机重排，保存到 `--shuffle-dir/{clip_key}_shuf.mp4`
   - `--make-composite`：拼接 `forward + black + reverse`，保存到 `--composite-dir/{clip_key}_t2v.mp4`
6. **编码加速**：所有 ffmpeg 输出使用 `libx264 -preset ultrafast -crf 23 -threads 1`
7. **可读性兜底**：每个输出文件自动做 `ffprobe` 验证，不可读时自动回退 `-preset medium` 重编码

输出 manifest 字段：`forward_video_path`, `reverse_video_path`, `composite_video_path`, `shuffle_video_path`, `clip_key`, `event_id`, `start_sec`, `end_sec`, `duration_sec`, `source_video_id`, `recipe_type`, `subset`, `sentence`, `sequence_index`, `shuffle_segment_sec`, `black_gap_sec`

### 2. `annotate_event_captions.py`

职责：为 manifest 中的 forward / reverse / shuffle 视频生成文本描述。

帧采样策略：

| 视频类型 | 采样方式 | 默认帧率 | 说明 |
|----------|----------|----------|------|
| forward | 按时长 × fps 均匀采样 | 1 fps | 足以描述整体动作方向 |
| reverse | 按时长 × fps 均匀采样 | 1 fps | 同上 |
| shuffle | segment-aware 均衡采样 | 2 fps | 与训练输入帧率对齐；每段至少 1 帧 |

核心参数：
- `--fwd-rev-fps 1.0`：forward/reverse 采样帧率
- `--shuffle-fps 2.0`：shuffle 采样帧率，与最终训练输入一致
- `--max-frames 32`：硬上限，防止超长视频产出过多帧

输出文件：
- `forward_captions.jsonl` / `reverse_captions.jsonl` / `shuffle_captions.jsonl`
- `caption_pairs.jsonl`（合并所有方向的 caption + confidence + direction_clear）

### 3. `build_aot_mcq.py`

职责：把 manifest 和 caption pairs 转成训练用 MCQ 数据。

过滤后构造三种 problem_type：
- `aot_v2t`：二选一，给视频选 caption（A/B）
- `aot_t2v`：二选一，给 caption 选 composite video 中前/后段（A/B）
- `aot_4way_v2t`：四选一，forward/reverse/shuffle/hard-negative（A/B/C/D）

### 4. `prompts.py`

集中维护的 prompt 模板：

| 函数 | 用途 |
|------|------|
| `SYSTEM_PROMPT` | 约束模型关注时序方向 |
| `get_forward_reverse_caption_prompt()` | 单段视频 caption 标注（含 `direction_clear` 判断） |
| `get_shuffle_caption_prompt(n_segments, segment_sec)` | shuffle 视频 caption 标注（段感知 prompt） |
| `get_v2t_prompt()` / `get_t2v_prompt()` | 最终 MCQ 问题模板（binary） |
| `get_4way_v2t_prompt()` | 最终 MCQ 问题模板（4-way） |

### 5. `rebalance_aot_answers.py`

在 offline / online filter 之后，对 `aot_v2t` / `aot_t2v` 样本重新平衡答案分布。
不丢样本，只交换部分多数类样本的选项顺序使 A/B 接近 1:1。

---

## 消融实验设计

核心问题：AoT proxy 数据以什么任务形式、什么选项粒度训练，才能最好地激发模型在下游时序理解任务上的潜力？

设计维度：

- **任务形式**（3 种）：V2T（给视频选描述）、T2V（给描述选视频）、Mixed（V2T + T2V 混合）
- **选项粒度**（2 种）：Binary（仅 forward vs reverse，A/B 两选一）、4-way（forward / reverse / shuffle / hard-negative，A/B/C/D 四选一）

3 × 2 = **6 组实验**，共用同一批标注数据，只改变送入训练的 problem_type 组合。

### 实验一览

| # | 实验名称 | 训练数据组成 | 选项数 | 核心假设 |
|---|---------|-------------|--------|---------|
| 1 | **V2T-Binary** | `aot_v2t` (forward vs reverse) | A/B | 最基础的正反方向区分，是否已足够 |
| 2 | **V2T-4way** | `aot_v2t` + `aot_4way_v2t` (forward / reverse / shuffle / hard-neg) | A/B/C/D | 增加 shuffle 和语义干扰项后，模型是否学到更精细的时序表征 |
| 3 | **T2V-Binary** | `aot_t2v` (给 caption 选 composite video 中正/反段) | A/B | 以文本为锚点、视频为候选，训练反向匹配能力 |
| 4 | **T2V-4way** | `aot_t2v` + `aot_4way_t2v` (正/反/shuffle/hard-neg video) | A/B/C/D | T2V 维度同样引入 shuffle 和干扰视频后的增益 |
| 5 | **Mixed-Binary** | `aot_v2t` + `aot_t2v`（均为 A/B） | A/B | V2T 和 T2V 双向训练，互为增强 |
| 6 | **Mixed-4way** | `aot_v2t` + `aot_t2v` + `aot_4way_v2t` + `aot_4way_t2v`（均含 A/B/C/D） | A/B/C/D | 全量组合，是否存在边际收益递减 |

所有实验均与 temporal_seg（YouCook2 时序分割）混合训练，seg 部分保持不变，仅改变 AoT proxy 部分。

### 实验 1：V2T-Binary

**训练数据**：仅 `aot_v2t`

- 给模型展示一个视频（forward 或 reverse）
- 提供 2 个 caption 选项（A: forward caption, B: reverse caption），选择与视频匹配的
- 这是最简单的 proxy 任务，考察模型能否区分正放和倒放的文本描述

**构造命令**：

```bash
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl data/aot_event_manifest.jsonl \
  --caption-pairs data/aot_annotations/caption_pairs.jsonl \
  --v2t-output data/exp1_v2t_binary.jsonl \
  --max-samples 1000
# 不传 --t2v-output 和 --fourway-output
```

### 实验 2：V2T-4way

**训练数据**：`aot_v2t` + `aot_4way_v2t`

- Binary V2T（A/B）和 4-way V2T（A/B/C/D）混合
- 4-way 的 4 个选项：forward caption / reverse caption / shuffle caption / 同 recipe 的 hard-negative caption
- shuffle caption 作为干扰项迫使模型区分「正确时序」vs「打乱时序」vs「完全不同」

**构造命令**：

```bash
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl data/aot_event_manifest.jsonl \
  --caption-pairs data/aot_annotations/caption_pairs.jsonl \
  --v2t-output data/exp2_v2t_binary.jsonl \
  --fourway-output data/exp2_v2t_4way.jsonl \
  --max-samples 1000
# 合并：cat data/exp2_v2t_binary.jsonl data/exp2_v2t_4way.jsonl > data/exp2_v2t_all.jsonl
```

### 实验 3：T2V-Binary

**训练数据**：仅 `aot_t2v`

- 给模型展示一段 caption，提供 2 个视频片段（composite video 中的前段 forward 和后段 reverse）
- 模型需要判断哪个视频与 caption 匹配
- 反向任务：从文本出发匹配视频，训练模型的 text→video 对齐能力

**构造命令**：

```bash
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl data/aot_event_manifest.jsonl \
  --caption-pairs data/aot_annotations/caption_pairs.jsonl \
  --t2v-output data/exp3_t2v_binary.jsonl \
  --max-samples 1000
# 不传 --v2t-output 和 --fourway-output
```

### 实验 4：T2V-4way

**训练数据**：`aot_t2v` + `aot_4way_t2v`

- Binary T2V（A/B）+ 4-way T2V（A/B/C/D）混合
- 4-way T2V 的 4 个视频选项：forward video / reverse video / shuffle video / 同 recipe 其他 clip 的 video
- 模型需从 4 个时序不同的视频中选出与 caption 精确匹配的

**构造命令**：

```bash
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl data/aot_event_manifest.jsonl \
  --caption-pairs data/aot_annotations/caption_pairs.jsonl \
  --t2v-output data/exp4_t2v_binary.jsonl \
  --fourway-t2v-output data/exp4_t2v_4way.jsonl \
  --max-samples 1000
# 合并：cat data/exp4_t2v_binary.jsonl data/exp4_t2v_4way.jsonl > data/exp4_t2v_all.jsonl
```

> 注：`--fourway-t2v-output` 为需要新增的参数，生成 T2V 方向的 4-way MCQ（给 caption 从 4 个视频中选）

### 实验 5：Mixed-Binary

**训练数据**：`aot_v2t` + `aot_t2v`（均为 A/B）

- V2T 和 T2V 双向 binary proxy 同时训练
- 假设：双向对齐比单向更有效，模型同时从 video→text 和 text→video 两个角度学习时序

**构造命令**：

```bash
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl data/aot_event_manifest.jsonl \
  --caption-pairs data/aot_annotations/caption_pairs.jsonl \
  --v2t-output data/exp5_v2t_binary.jsonl \
  --t2v-output data/exp5_t2v_binary.jsonl \
  --max-samples 1000
# 合并：cat data/exp5_v2t_binary.jsonl data/exp5_t2v_binary.jsonl > data/exp5_mixed_binary.jsonl
```

### 实验 6：Mixed-4way

**训练数据**：`aot_v2t` + `aot_t2v` + `aot_4way_v2t` + `aot_4way_t2v`（全量组合）

- 双向 × 两种粒度的四合一训练
- 假设：完整覆盖所有 proxy 变体能最大化时序理解能力
- 但也可能存在边际收益递减或训练信号冲突

**构造命令**：

```bash
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl data/aot_event_manifest.jsonl \
  --caption-pairs data/aot_annotations/caption_pairs.jsonl \
  --v2t-output data/exp6_v2t_binary.jsonl \
  --t2v-output data/exp6_t2v_binary.jsonl \
  --fourway-output data/exp6_v2t_4way.jsonl \
  --fourway-t2v-output data/exp6_t2v_4way.jsonl \
  --max-samples 1000
# 合并所有
```

### 评估体系

所有 6 组实验共用同一个评估框架：

| 评估维度 | 指标 | 说明 |
|----------|------|------|
| **AoT proxy 内部** | V2T accuracy, T2V accuracy | held-out 验证集上的 MCQ 准确率 |
| **下游时序理解** | temporal ordering accuracy | 在独立的时序排序 benchmark 上评估泛化能力 |
| **训练效率** | reward 收敛速度 | 达到相同 accuracy 所需的训练步数 |
| **答案偏置** | A/B/C/D 选择率分布 | 是否存在位置偏好（尤其 binary 的 A/B 偏置） |
| **代价** | 数据构造时间 | 4-way 需要额外的 shuffle 标注 + hard-negative 匹配 |

### 关键对比

预期最有价值的对比方向：

```
实验 1 vs 实验 2   → shuffle 干扰项对 V2T 的增益（binary → 4-way 的边际价值）
实验 3 vs 实验 4   → 同上，在 T2V 方向的增益
实验 1 vs 实验 3   → V2T vs T2V 哪个方向更有效
实验 5 vs 实验 1   → 混合双向 vs 单向 V2T 的增益
实验 5 vs 实验 6   → binary 混合 vs 4-way 混合的增益（是否 4-way 对混合训练也有帮助）
实验 2 vs 实验 6   → 单向 4-way V2T vs 全量组合（边际收益递减点在哪里）
```

### 控制变量

为保证实验可比性：

1. **数据量对齐**：所有实验的 AoT proxy 样本总数保持一致（如均 1000 条）；4-way 实验中 binary + 4-way 样本按 1:1 混合，总量不变
2. **temporal_seg 固定**：YouCook2 时序分割数据在所有实验中完全相同
3. **训练超参一致**：学习率、batch size、epoch 数、模型初始化均一致
4. **答案重平衡**：所有实验均做 rebalance，消除答案分布差异的混淆因素
5. **随机种子**：固定 3 组种子，每组实验跑 3 次取均值

### 运行顺序

6 组实验共享同一批标注数据（Step 1-2 只需跑一次），仅 Step 3 的 `build_aot_mcq.py` 参数不同：

```
Step 1：build_event_aot_data.py（生成 manifest + reverse/shuffle/composite 素材）  ← 跑一次
Step 2：annotate_event_captions.py（标注所有方向的 caption）                        ← 跑一次
Step 3：build_aot_mcq.py × 6 次（不同 --output 参数组合）                          ← 并行
Step 4：mix_aot_with_youcook2.py × 6 次（各自混合 temporal_seg）                   ← 并行
Step 5：训练 × 6 组 × 3 seeds = 18 次                                             ← GPU 并行
Step 6：统一评估                                                                    ← 并行
```

---

## 依赖

```bash
pip install openai pillow decord tqdm
```

另外需要：

- `ffmpeg` / `ffprobe`
- 一个 OpenAI-compatible VLM endpoint

---

## 详细用法

### 1. 生成 event manifest + reverse 素材

最常用的跑法是：直接读现有 `extracted_clips_database.json`，并生成 reverse。

```bash
python proxy_data/temporal_aot/build_event_aot_data.py \
  --clip-db-json /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/extracted_clips_database.json \
  --subset training \
  --output-jsonl proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --reverse-dir /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/reverse_clips \
  --invalid-report-jsonl proxy_data/temporal_aot/data/aot_invalid_clips.jsonl \
  --make-reverse \
  --max-samples 1000 \
  --min-duration 3 \
  --max-duration-diff-sec 2.0
```

复用坏样本列表跳过全量 validation：

```bash
python proxy_data/temporal_aot/build_event_aot_data.py \
  --clip-db-json /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/extracted_clips_database.json \
  --subset training \
  --output-jsonl proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --reverse-dir /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/reverse_clips \
  --invalid-report-jsonl proxy_data/temporal_aot/data/aot_invalid_clips.jsonl \
  --bad-samples-jsonl proxy_data/temporal_aot/data/aot_invalid_clips.jsonl \
  --make-reverse \
  --max-samples 1000 \
  --min-duration 3 \
  --max-duration-diff-sec 2.0
```

全量生成（reverse + shuffle + composite）：

```bash
python proxy_data/temporal_aot/build_event_aot_data.py \
  --clip-db-json /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/extracted_clips_database.json \
  --subset training \
  --output-jsonl /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/data/aot_event_manifest.jsonl \
  --reverse-dir /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/reverse_clips \
  --shuffle-dir /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/shuffle_clips \
  --composite-dir /m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/composite_clips \
  --make-reverse \
  --make-shuffle \
  --make-composite \
  --shuffle-segment-sec 2.0 \
  --min-shuffle-segments 3 \
  --max-samples 4000 \
  --min-duration 3 \
  --build-workers 8
```

关键参数说明：
- `--subset`：只在 `--clip-db-json` 模式下生效
- `--max-samples`：控制最终参与 AoT 构造的 event clip 数量
- `--min-duration`：过滤过短事件
- `--max-duration-diff-sec`：过滤标注时长与实际时长差异过大的 clip
- `--build-workers`：并行 ffmpeg 工作线程数
- `--invalid-report-jsonl`：记录被过滤/失败的样本，可回灌给 `--bad-samples-jsonl`

### 2. 标注 forward / reverse / shuffle caption

先配置 API key：

```bash
export NOVITA_API_KEY=your_novita_api_key
# 或
export OPENAI_API_KEY=your_api_key
```

```bash
python proxy_data/temporal_aot/annotate_event_captions.py \
  --manifest-jsonl proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --output-dir proxy_data/temporal_aot/data/aot_annotations \
  --api-base https://api.novita.ai/v3/openai \
  --model pa/gmn-2.5-pr \
  --workers 4 \
  --fwd-rev-fps 1.0 \
  --shuffle-fps 2.0 \
  --max-frames 32 \
  --max-samples 500
```

输出文件：
- `forward_captions.jsonl` / `reverse_captions.jsonl` / `shuffle_captions.jsonl`
- `caption_pairs.jsonl`

### 3. 构造 V2T / T2V / 4-way 数据集

```bash
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --caption-pairs proxy_data/temporal_aot/data/aot_annotations/caption_pairs.jsonl \
  --v2t-output proxy_data/temporal_aot/data/aot_annotations/v2t_train.jsonl \
  --t2v-output proxy_data/temporal_aot/data/aot_annotations/t2v_train.jsonl \
  --fourway-output proxy_data/temporal_aot/data/aot_annotations/4way_train.jsonl \
  --max-samples 500 \
  --min-confidence 0.6
```

### 4. 混合 YouCook2 + AoT 训练/验证集

```bash
python proxy_data/temporal_aot/mix_aot_with_youcook2.py \
  --seg-jsonl proxy_data/youcook2_train_easyr1.jsonl \
  --v2t-jsonl proxy_data/temporal_aot/data/aot_annotations/v2t_train.jsonl \
  --t2v-jsonl proxy_data/temporal_aot/data/aot_annotations/t2v_train.jsonl \
  --train-output proxy_data/temporal_aot/data/mixed_aot_train.jsonl \
  --val-output proxy_data/temporal_aot/data/mixed_aot_val.jsonl \
  --train-per-source 400 \
  --val-per-source 30 \
  --seed 42
```

### 5. 答案重平衡

```bash
python proxy_data/temporal_aot/rebalance_aot_answers.py \
  --input-jsonl proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.jsonl \
  --output-jsonl proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.balanced.jsonl
```

按 problem_type 统一打平：

```bash
python proxy_data/temporal_aot/rebalance_aot_answers.py \
  --input-jsonl proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.jsonl \
  --output-jsonl /tmp/mixed_aot_train.balanced.jsonl \
  --balance-scope all
```

---

## 注意事项

1. `annotate_event_captions.py` 在缺少 `reverse_video_path` 时，会回退到 forward 视频继续标注
2. `build_aot_mcq.py` 只有在 manifest 存在 `composite_video_path` 时才会产出 `aot_t2v`
3. 去重只基于路径，不基于视频内容哈希
4. ultrafast 编码在绝大多数场景下可读，但对码率极低或分辨率极端的视频可能产出问题——脚本已内置自动兜底
5. 推荐最终训练输入是 `extracted_clips_database.json`，因为它已提供 `clip_path`、`sentence`、`subset`、`sequence_index` 等现成标注
