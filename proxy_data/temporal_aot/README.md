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

1. `build_event_aot_data.py`
   读取 `extracted_clips_database.json`，按 `clip_path` 去重，生成 manifest；按需生成 `reverse`；按需生成 `composite`。
2. `annotate_event_captions.py`
   读取 manifest，对 `forward` / `reverse` 视频分别抽帧并调用 VLM，得到 caption 和 confidence。
3. `build_aot_mcq.py`
   读取 manifest 和 caption pairs，筛掉低置信度或正反描述没有差异的样本，构造 `aot_v2t` / `aot_t2v` 训练数据。
4. `prompts.py`
   统一维护标注和 MCQ 用的 prompt 模板。

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
   - 若能得到时长，则 `duration_sec >= --min-duration`
   - 文件必须真实存在
   - 如果指定 `--subset training` 或 `--subset validation`，只保留对应子集
5. 打乱顺序后，按 `--max-samples` 截断。
6. 如果指定 `--make-reverse`：
   - 用 `ffmpeg -vf reverse -an` 生成 `reverse`
   - 保存到 `--reverse-dir/{clip_key}_rev.mp4`
7. 如果指定 `--make-composite`：
   - 确保 `reverse` 已存在
   - 生成或复用黑屏视频
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
pip install openai pillow decord
```

另外需要：

- `ffmpeg`
- 一个 OpenAI-compatible VLM endpoint

## 1. 生成 event manifest + reverse 素材

最常用的跑法是：直接读现有 `extracted_clips_database.json`，并生成 reverse。

```bash
python proxy_data/temporal_aot/build_event_aot_data.py \
  --clip-db-json /m2v_intern/xuboshen/zgw/data/youcook2_event_clips/extracted_clips_database.json \
  --subset training \
  --output-jsonl /path/to/aot_event_manifest.jsonl \
  --reverse-dir /path/to/reverse_clips \
  --make-reverse \
  --max-samples 1000 \
  --min-duration 3
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
  --output-jsonl /path/to/aot_event_manifest.jsonl \
  --reverse-dir /path/to/reverse_clips \
  --composite-dir /path/to/t2v_composite \
  --make-reverse \
  --make-composite \
  --black-gap-sec 2.0 \
  --max-samples 1000 \
  --min-duration 3
```

兼容旧流程时，也可以继续传 `--input-jsonl`，但不再是推荐方式。

说明：

- `--subset` 只在 `--clip-db-json` 模式下生效
- `--max-samples` 控制最终参与 AoT 构造的 event clip 数量
- `--min-duration` 用于过滤过短事件
- 这一步已经完成路径级去重
- 如果不加 `--make-reverse`，manifest 里的 `reverse_video_path` 会为空

## 2. 标注 forward / reverse caption

```bash
python proxy_data/temporal_aot/annotate_event_captions.py \
  --manifest-jsonl /path/to/aot_event_manifest.jsonl \
  --output-dir /path/to/aot_annotations \
  --api-base http://localhost:8000/v1 \
  --model Qwen3-VL-7B \
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
  --manifest-jsonl /path/to/aot_event_manifest.jsonl \
  --caption-pairs /path/to/aot_annotations/caption_pairs.jsonl \
  --v2t-output /path/to/v2t_train.jsonl \
  --t2v-output /path/to/t2v_train.jsonl \
  --max-samples 500 \
  --min-confidence 0.6
```

输出：

- `v2t_train.jsonl`
- `t2v_train.jsonl`

其中：

- `problem_type = aot_v2t`
- `problem_type = aot_t2v`

## 4. 推荐启动顺序

1. 先直接用 `extracted_clips_database.json` 跑出一批 manifest
2. 打开 `--make-reverse`，确认 reverse 视频可用
3. 再做 caption 标注，先检查 `caption_pairs.jsonl`
4. 先构造 `V2T`
5. 只有确认需要 `T2V` 时，再生成 composite

## 5. 当前实现的注意点

1. `annotate_event_captions.py` 在缺少 `reverse_video_path` 时，会回退到 forward 视频继续标注。
2. `build_aot_mcq.py` 只有在 manifest 存在 `composite_video_path` 时才会产出 `aot_t2v`。
3. 去重只基于路径，不基于视频内容哈希。
4. 当前最推荐的输入是 `extracted_clips_database.json`，因为它已经提供了 `clip_path`、`sentence`、`subset`、`sequence_index` 等现成标注。
