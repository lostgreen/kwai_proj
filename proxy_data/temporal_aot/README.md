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

针对 AoT 标注流程中的关键设计选择，设计以下六组消融实验，逐一验证各组件对最终训练效果的贡献：

### 实验一览

| # | 实验名称 | 变量 | 对照组 (baseline) | 实验组 | 评估指标 |
|---|---------|------|-------------------|--------|----------|
| 1 | **Shuffle 标注帧率** | `--shuffle-fps` | 1 fps（与 fwd/rev 一致） | 2 fps（与训练输入对齐） | shuffle caption 质量、4-way MCQ 准确率 |
| 2 | **Shuffle Prompt 段感知** | prompt 中是否告知段数和段长 | 旧 prompt（只说 "short segments"） | 新 prompt（告知 N segments × 2s） | shuffle caption 的段覆盖率、与 forward caption 的 ROUGE-L 差异 |
| 3 | **Direction-clear 过滤** | `--require-direction-clear` | 不过滤（`--no-require-direction-clear`） | 过滤 direction_clear=False 的样本 | V2T 训练收敛速度、最终准确率 |
| 4 | **4-way vs Binary V2T** | `problem_type` | 仅 `aot_v2t` (A/B) | `aot_v2t` + `aot_4way_v2t` (A/B/C/D) | 时序理解准确率、泛化性 |
| 5 | **Hard-negative 来源** | 4-way 的 D 选项来自哪里 | 随机 caption（任意 recipe） | 同 recipe_type 的 caption（语义相近干扰） | 4-way MCQ 难度、模型细粒度区分能力 |
| 6 | **答案重平衡** | 是否做 rebalance | 不做（自然分布） | 做 rebalance（A/B ≈ 1:1） | 答案偏置程度、训练稳定性 |

### 实验 1：Shuffle 标注帧率

**动机**：训练阶段视频以 2fps 输入模型，如果 VLM 标注时只用 1fps，则每个 2s 段只有 2 帧，可能丢失关键帧间变化。

| 配置 | 命令差异 |
|------|---------|
| baseline | `--shuffle-fps 1.0 --fwd-rev-fps 1.0` |
| experiment | `--shuffle-fps 2.0 --fwd-rev-fps 1.0` |

**评估方式**：
1. 对相同 100 个 shuffle clip，分别用 1fps 和 2fps 标注 caption
2. 人工评估 caption 中是否正确描述了每个 2s 段的内容
3. 以两组 caption 分别构造 4-way MCQ，用同一个 VLM 做 zero-shot 测试比较准确率

### 实验 2：Shuffle Prompt 段感知

**动机**：VLM 不知道视频被切成了几段、每段多长，可能输出泛泛描述（"food is being prepared"）而非逐段描述。

| 配置 | Prompt 差异 |
|------|------------|
| baseline | "cut into short segments and randomly reordered" |
| experiment | "cut into 5 segments of approximately 2 seconds each and randomly reordered" |

**评估方式**：
1. 统计 caption 中显式时序标记词（first/then/next/finally）的出现次数
2. 计算 shuffle caption 与 forward caption 的 ROUGE-L，越低越好（说明描述了不同的东西）
3. 对「无明显变化」的段，检查是否出现了类似 "remains the same" 的标注

### 实验 3：Direction-clear 过滤

**动机**：搅拌、揉面等循环动作正反播放视觉几乎一样，保留这些样本会引入噪声标签。

| 配置 | 过滤策略 |
|------|---------|
| baseline | `--no-require-direction-clear`（保留所有样本） |
| experiment | 默认（过滤 `direction_clear=False`） |

**评估方式**：
1. 比较两组的 V2T 训练 loss 曲线
2. 在验证集上比较最终 V2T/T2V 准确率
3. 统计过滤掉的样本中有多少确实是循环动作（人工抽检 50 个）

### 实验 4：4-way vs Binary V2T

**动机**：4-way MCQ 包含 shuffle 和 hard-negative 干扰项，理论上能迫使模型学到更精细的时序表征。

| 配置 | 训练数据 |
|------|---------|
| baseline | 仅 `aot_v2t` (binary A/B) + `temporal_seg` |
| experiment | `aot_v2t` + `aot_4way_v2t` + `temporal_seg`，总量对齐 |

**评估方式**：
1. 在 held-out 的 binary V2T 验证集上比较准确率
2. 在 held-out 的 4-way V2T 验证集上比较准确率
3. 比较模型对 shuffle video 的区分能力（shuffle vs forward 配错率）

### 实验 5：Hard-negative 来源

**动机**：来自同 recipe_type 的 hard-negative 语义更接近正确选项，能测试模型是否真正理解了时序而非仅靠语义匹配。

| 配置 | D 选项来源 |
|------|-----------|
| baseline | 随机 recipe 的 caption |
| experiment | 同 recipe_type 的其他 clip 的 forward caption |

**评估方式**：
1. 比较两组 4-way MCQ 的平均正确率
2. 分析错误分布：实验组是否更多地把 D（hard-negative）误选为答案
3. 在下游时序推理 benchmark 上比较泛化效果

### 实验 6：答案重平衡

**动机**：过滤后可能出现 A/B 分布偏斜（如 A:70% / B:30%），模型可能学到答案偏置。

| 配置 | 策略 |
|------|------|
| baseline | 直接用过滤后的数据训练 |
| experiment | 先做 `rebalance_aot_answers.py` 再训练 |

**评估方式**：
1. 统计模型在验证集上的 A 选择率和 B 选择率
2. 比较两组的 V2T accuracy（特别关注原始少数类的 recall）
3. 分析训练 loss 方差

### 消融实验运行顺序

推荐按依赖关系分批运行：

```
第一批（并行）：实验 1 + 实验 2    ← 只影响标注阶段
第二批（并行）：实验 3 + 实验 6    ← 只影响过滤/重平衡阶段
第三批（并行）：实验 4 + 实验 5    ← 只影响数据构造/训练阶段
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
