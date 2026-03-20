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
4. 如需 `T2V`，输入视频仍然是 `forward + black gap + reverse`

## 整体流程

按顺序依次运行以下脚本：

**Step 1 — 生成 manifest + reverse/shuffle/composite 视频素材**

```bash
python proxy_data/temporal_aot/build_event_aot_data.py \
  --clip-db-json /path/to/extracted_clips_database.json \
  --output-jsonl proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --reverse-dir /path/to/reverse_clips \
  --shuffle-dir /path/to/shuffle_clips \
  --make-reverse \
  --make-shuffle \
  --shuffle-segment-sec 2.0 \
  --min-shuffle-segments 3 \
  --build-workers 8 \
  --invalid-report-jsonl proxy_data/temporal_aot/data/aot_invalid_clips.jsonl \
  --subset training
```

说明：
- `--build-workers 8`：并行 ffmpeg，可大幅缩短生成时间（原串行约 6s/clip，并行后预计 ≤1s/clip）
- `--make-shuffle`：对 duration ≥ `min-shuffle-segments × shuffle-segment-sec`（默认 6s）的 clip 生成对应的时间乱序视频，保存为 `{clip_key}_shuf.mp4`
- 不满足最小分段数的 clip 会跳过 shuffle 生成，仍正常写入 manifest（`shuffle_video_path` 为空字符串）

**Step 2 — VLM 标注 caption + direction_clear**

```bash
python proxy_data/temporal_aot/annotate_event_captions.py \
  --manifest-jsonl proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --output-dir proxy_data/temporal_aot/data/aot_annotations \
  --api-base http://localhost:8000/v1 \
  --model Qwen3-VL-7B \
  --workers 8
```

VLM 会对每个 clip 输出 `caption`、`confidence`、`direction_clear` 三个字段。
`direction_clear=false` 表示该动作（如搅拌、揉面）在倒放后外观几乎相同，VLM 自动标记为方向不明显，后续会被过滤。

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

1. `build_event_aot_data.py`
   读取 `extracted_clips_database.json`，按 `clip_path` 去重，生成 manifest；按需生成 `reverse`；按需生成 `composite`。
2. `annotate_event_captions.py`
   读取 manifest，对 `forward` / `reverse` 视频分别抽帧并调用 VLM，得到 `caption`、`confidence`、`direction_clear`。
3. `build_aot_mcq.py`
   读取 manifest 和 caption pairs，过滤低置信度、正反无差异、以及 VLM 判断方向不明显的样本，构造 `aot_v2t` / `aot_t2v` 训练数据。
4. `mix_aot_with_youcook2.py`
   把 V2T / T2V 与 YouCook2 时序分割数据合并为统一 train/val split。
5. `prompts.py`
   统一维护标注和 MCQ 用的 prompt 模板（含 `direction_clear` 判断指令）。
6. `rebalance_aot_answers.py`
   在 offline filter 之后对保留的 AOT 样本做 A/B 选项重排，把最终答案分布尽量打平，不丢样本。

## 脚本逻辑

### 1. `build_event_aot_data.py`

职责：从现成 event clip 标注构造 AoT manifest，并补充 reverse/composite 素材。

推荐输入：

- `--clip-db-json /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/extracted_clips_database.json`

兼容输入：

- `--input-jsonl proxy_data/proxy_train_easyr1.jsonl`

实际逻辑：

1. 二选一读取输入源：
   - 推荐：`--clip-db-json`
   - 兼容：`--input-jsonl`
2. 如果读取 `clip-db-json`：
   - 遍历 `video_id -> list[clip_info]`
   - 取每个 `clip_info["clip_path"]`
   - 合并已有标注字段：
     - `original_video_id`
     - `recipe_type`
     - `subset`
     - `sentence`
     - `segment_in_original`
     - `event_id`
     - `sequence_index`
3. 如果文件名符合 `*_event{event_id}_{start}_{end}.mp4`，脚本也会解析出：
   - `clip_key`
   - `start_sec`
   - `end_sec`
   - `duration_sec`
4. 过滤条件：
   - `clip_path` / `video_path` 必须是字符串
   - 同一个路径只保留一次
   - 如果指定 `--bad-samples-jsonl`，会先按 `clip_key` / `video_path` 跳过已知坏样本，并跳过前置 `ffprobe` validation
   - 若能得到时长，则 `duration_sec >= --min-duration`
   - 文件必须真实存在
   - 如果指定 `--subset training` 或 `--subset validation`，只保留对应子集
   - 会用 `ffprobe` 检查视频是否可读、是否存在 video stream
   - 如果文件名/标注里的事件时长与实际视频时长相差超过 `--max-duration-diff-sec`，会直接过滤
5. 打乱顺序后，按 `--max-samples` 截断。
6. 如果指定 `--make-reverse`：
   - 用 `ffmpeg -vf reverse -an` 生成 `reverse`
   - 保存到 `--reverse-dir/{clip_key}_rev.mp4`
7. 如果指定 `--make-composite`：
   - 确保 `reverse` 已存在
   - 生成或复用黑屏视频
   - black gap 会在拼接前自动对齐到当前样本的分辨率和帧率
   - 拼接 `forward + black + reverse`
   - 保存到 `--composite-dir/{clip_key}_t2v.mp4`
8. 输出 manifest，每条记录包含：
   - `forward_video_path`
   - `reverse_video_path`
   - `composite_video_path`
   - `clip_key`
   - `event_id`
   - `start_sec`
   - `end_sec`
   - `duration_sec`
   - `source_video_id`
   - `recipe_type`
   - `subset`
   - `sentence`
   - `sequence_index`

补充说明：

- 这里不再导出新的 forward 视频，`forward_video_path` 直接引用现有 `clip_path`。
- 真正新增的离线素材主要是 `reverse`，这也是当前最需要补的部分。
- 如果后面要做 `aot_t2v`，仍然可以额外打开 `--make-composite`。
- 如果某个视频虽然存在，但 `ffprobe`/`ffmpeg` 读不出来，脚本现在会跳过该样本，不再让整批任务直接报错退出。
- 可以通过 `--invalid-report-jsonl` 把被过滤/失败的样本和原因写出来，方便回查。
- `--invalid-report-jsonl` 输出的 JSONL 也可以直接回灌给 `--bad-samples-jsonl`，下次运行时会跳过这些样本，并且不再做全量前置 validation。

### 2. `annotate_event_captions.py`

职责：为 manifest 中的 `forward` / `reverse` 视频生成文本描述。

实际逻辑：

1. 读取 manifest JSONL。
2. 对每条记录：
   - 取 `forward_video_path`
   - 取 `reverse_video_path`，如果为空，则退回到 `forward_video_path`
3. 用 `decord` 对每个视频均匀抽帧。
4. 把抽出的帧压成 JPEG data URL。
5. 对 forward / reverse 各请求一次 VLM。
6. 输出：
   - `forward_captions.jsonl`
   - `reverse_captions.jsonl`
   - `caption_pairs.jsonl`

注意：

- 这个脚本本身不做去重，默认上游 manifest 已经去重。
- 如果上游没有真的生成 reverse，脚本会回退到 forward 视频继续标注，所以建议至少先跑 `--make-reverse`。
- 输出文件会按样本逐行写入；如果中途手动中断，已经完成的结果会保留在 JSONL 里。

### 3. `build_aot_mcq.py`

职责：把 manifest 和 caption pairs 转成训练用 MCQ 数据。

实际逻辑：

1. 读取 manifest，并用 `clip_key` 建索引。
2. 读取 `caption_pairs.jsonl` 并打乱顺序。
3. 过滤掉：
   - `forward_confidence < --min-confidence`
   - `reverse_confidence < --min-confidence`
   - `is_different == False`
   - manifest 中找不到 `clip_key`
4. 构造 `aot_v2t`：
   - forward 视频对应答案 `A`
   - reverse 视频对应答案 `B`
5. 如果 manifest 里有 `composite_video_path`，再构造 `aot_t2v`：
   - `forward_caption -> A`
   - `reverse_caption -> B`

### 4. `prompts.py`

职责：集中维护 prompt。

- `SYSTEM_PROMPT`：约束模型关注时序方向
- `get_forward_reverse_caption_prompt()`：单段视频 caption 标注
- `get_v2t_prompt()` / `get_t2v_prompt()`：最终 MCQ 问题模板

### 5. `rebalance_aot_answers.py`

职责：在 offline / online filter 之后，对最终保留下来的 `aot_v2t` / `aot_t2v` 样本重新平衡答案分布。

实际逻辑：

1. 读取过滤后的 mixed JSONL。
2. 仅处理 `aot_t2v` / `aot_v2t`。
3. 找到当前多数答案（`A` 或 `B`）。
4. 随机挑选一部分多数类样本，交换选项顺序：
   - `aot_v2t`：交换 caption 的 `A/B` 顺序
   - `aot_t2v`：交换 `The first segment` / `The second segment` 的 `A/B` 顺序
5. 同步翻转 `answer`，并在 `metadata` 里写入重平衡标记。
6. 输出样本数不变，只让 A/B 尽量接近 `1:1`。

最常用的跑法：

```bash
python proxy_data/temporal_aot/rebalance_aot_answers.py \
  --input-jsonl proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.jsonl \
  --output-jsonl proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.balanced.jsonl
```

如果想把所有 AOT 样本放在一起统一打平，而不是按 `problem_type` 分开打平：

```bash
python proxy_data/temporal_aot/rebalance_aot_answers.py \
  --input-jsonl proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.jsonl \
  --output-jsonl /tmp/mixed_aot_train.balanced.jsonl \
  --balance-scope all
```

## 检查结论

### 是否对视频片段做了去重

做了，当前是按路径精确去重。

- 去重位置：`build_event_aot_data.py`
- 去重键：
  - 读 `clip-db-json` 时是 `clip_path`
  - 读 `input-jsonl` 时是 `video_path`

这意味着：

- 同一条 clip 在标注文件里重复出现，只保留第一次。
- 内容相同但路径不同的两个视频，当前不会做内容级去重。

### 是否保存了 forward 和 reverse 两个片段

当前实现是：

- `forward`
  - 直接复用现有切好的 clip
  - 在 manifest 里记录为 `forward_video_path`
  - 不会再额外复制一份
- `reverse`
  - 通过 `--make-reverse` 或 `--make-composite` 生成
  - 保存到 `--reverse-dir/{clip_key}_rev.mp4`

所以现在最准确的表述是：

- forward 已经存在，直接复用
- reverse 需要脚本额外生成并保存

这也正好符合你现在这份 `youcook2_event_clips` 数据的使用方式。

## 0. 依赖

```bash
pip install openai pillow decord tqdm
```

另外需要：

- `ffmpeg`
- 一个 OpenAI-compatible VLM endpoint
- `build_event_aot_data.py` 运行时会显示 `tqdm` 进度条
- `annotate_event_captions.py` 运行时也会显示 `tqdm` 进度条

## 1. 生成 event manifest + reverse 素材

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

如果想复用上次记录下来的坏样本列表，避免再做一遍全量 `ffprobe` validation：

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

如果只想先生成 manifest，不落 reverse：

```bash
python proxy_data/temporal_aot/build_event_aot_data.py \
  --clip-db-json /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/extracted_clips_database.json \
  --subset training \
  --output-jsonl /path/to/aot_event_manifest.jsonl \
  --max-samples 1000 \
  --min-duration 3
```

如果还要构造 `T2V composite`：

```bash
python proxy_data/temporal_aot/build_event_aot_data.py \
  --clip-db-json /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/extracted_clips_database.json \
  --subset training \
  --output-jsonl proxy_data/temporal_aot/data/aot_event_manifest_all.jsonl \
  --reverse-dir /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/reverse_clips \
  --composite-dir /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/composite_clips \
  --make-reverse \
  --make-composite \
  --max-samples 1000 \
  --min-duration 3

```

兼容旧流程时，也可以继续传 `--input-jsonl`，但不再是推荐方式。

说明：

- `--subset` 只在 `--clip-db-json` 模式下生效
- `--max-samples` 控制最终参与 AoT 构造的 event clip 数量
- `--min-duration` 用于过滤过短事件
- `--max-duration-diff-sec` 用于过滤“文件名/标注时长”和“真实视频时长”差太多的 clip
- `--invalid-report-jsonl` 会记录被过滤或 reverse/composite 生成失败的样本
- `--bad-samples-jsonl` 可直接复用历史 `invalid-report`，提前跳过这些样本，并关闭全量前置 validation
- 这一步已经完成路径级去重
- 如果不加 `--make-reverse`，manifest 里的 `reverse_video_path` 会为空

## 2. 标注 forward / reverse caption

先配置 API key。你可以二选一：

```bash
export NOVITA_API_KEY=your_novita_api_key
```

或者：

```bash
export OPENAI_API_KEY=your_api_key
```

如果不想配环境变量，也可以在命令行里直接传 `--api-key xxx`。

```bash
python proxy_data/temporal_aot/annotate_event_captions.py \
  --manifest-jsonl /home/xuboshen/zgw/EasyR1/proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --output-dir /home/xuboshen/zgw/EasyR1/proxy_data/temporal_aot/data/aot_annotations \
  --api-base https://api.novita.ai/v3/openai \
  --model pa/gmn-2.5-pr \
  --workers 4 \
  --max-frames 16 \
  --max-samples 500
```

输出：

- `/path/to/aot_annotations/forward_captions.jsonl`
- `/path/to/aot_annotations/reverse_captions.jsonl`
- `/path/to/aot_annotations/caption_pairs.jsonl`

## 3. 构造 V2T / T2V 数据集

```bash
python proxy_data/temporal_aot/build_aot_mcq.py \
  --manifest-jsonl /home/xuboshen/zgw/EasyR1/proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --caption-pairs /home/xuboshen/zgw/EasyR1/proxy_data/temporal_aot/data/aot_annotations/caption_pairs.jsonl \
  --v2t-output /home/xuboshen/zgw/EasyR1/proxy_data/temporal_aot/data/aot_annotations/v2t_train.jsonl \
  --t2v-output /home/xuboshen/zgw/EasyR1/proxy_data/temporal_aot/data/aot_annotations/t2v_train.jsonl \
  --max-samples 500 \
  --min-confidence 0.6
```

输出：

- `v2t_train.jsonl`
- `t2v_train.jsonl`

其中：

- `problem_type = aot_v2t`
- `problem_type = aot_t2v`

## 4. 混合 YouCook2 + AoT 训练/验证集

如果你想把：

- `proxy_data/youcook2_train_easyr1.jsonl`
- `v2t_train.jsonl`
- `t2v_train.jsonl`

混成一个新的训练集/验证集，可以直接用：

```bash
python proxy_data/temporal_aot/mix_aot_with_youcook2.py \
  --seg-jsonl proxy_data/youcook2_train_easyr1.jsonl \
  --v2t-jsonl /path/to/v2t_train.jsonl \
  --t2v-jsonl /path/to/t2v_train.jsonl \
  --train-output proxy_data/temporal_aot/data/mixed_aot_train.jsonl \
  --val-output proxy_data/temporal_aot/data/mixed_aot_val.jsonl \
  --train-per-source 400 \
  --val-per-source 30 \
  --seed 42
```

默认逻辑：

- `youcook2_train_easyr1.jsonl` 抽 400 条 train，30 条 val
- `v2t_train.jsonl` 抽 400 条 train，30 条 val
- `t2v_train.jsonl` 抽 400 条 train，30 条 val
- 每个来源内部 train / val 不重叠
- 最终再分别打乱，输出混合后的 train / val
- 如果 `youcook2_train_easyr1.jsonl` 的 `problem_type` 为空，会自动补成 `temporal_seg`
- 输出时会额外写入 `metadata.mix_source`，方便后续检查样本来源

## 5. 推荐启动顺序

1. 先直接用 `extracted_clips_database.json` 跑出一批 manifest
2. 打开 `--make-reverse`，确认 reverse 视频可用
3. 再做 caption 标注，先检查 `caption_pairs.jsonl`
4. 先构造 `V2T`
5. 只有确认需要 `T2V` 时，再生成 composite

## 6. 当前实现的注意点

1. `annotate_event_captions.py` 在缺少 `reverse_video_path` 时，会回退到 forward 视频继续标注。
2. `build_aot_mcq.py` 只有在 manifest 存在 `composite_video_path` 时才会产出 `aot_t2v`。
3. 去重只基于路径，不基于视频内容哈希。
4. 当前最推荐的输入是 `extracted_clips_database.json`，因为它已经提供了 `clip_path`、`sentence`、`subset`、`sequence_index` 等现成标注。
