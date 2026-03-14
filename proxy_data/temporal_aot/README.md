# Temporal AoT Scripts

这个目录是一条“事件级 Temporal AoT”数据管线。代码里把单个 `event clip` 作为最小处理单元，不再回到长视频窗口。

核心假设：

1. 以单个 event clip 为基本单位
2. `reverse` 指该 event clip 的帧级倒放
3. `T2V` 的输入视频是 `forward clip + black gap + reverse clip`

所有输入路径和输出路径都通过命令行参数指定。

## 整体流程

1. `build_event_aot_data.py`
   从 proxy JSONL 里收集 event clip，做过滤和去重，生成 manifest；可选导出 `reverse` 片段；可选拼接 `composite` 视频。
2. `annotate_event_captions.py`
   读取 manifest，对 `forward` / `reverse` 视频分别抽帧并调用 VLM，得到 caption 和 confidence。
3. `build_aot_mcq.py`
   读取 manifest 和 caption pairs，筛掉低置信度或正反描述没有差异的样本，构造 `aot_v2t` / `aot_t2v` 训练数据。
4. `prompts.py`
   统一维护 caption 标注和 MCQ 构造所用的 prompt 模板。

## 脚本逻辑

### 1. `build_event_aot_data.py`

职责：构造事件级 manifest，并按需生成离线视频素材。

实际逻辑：

1. 逐行读取输入 JSONL。
2. 从每条样本的 `videos` 字段中取出视频路径。
3. 用文件名模式 `*_event{event_id}_{start}_{end}.mp4` 解析：
   - `clip_key`
   - `event_id`
   - `start_sec`
   - `end_sec`
   - `duration_sec`
   - `source_video_id`
4. 过滤条件：
   - `video_path` 必须是字符串
   - 同一个 `video_path` 只保留一次
   - 若能解析出时长，则 `duration_sec >= --min-duration`
   - 文件必须真实存在
5. 收集额外元信息：
   - `metadata.recipe_type`
   - `metadata.task_type`
6. 打乱顺序后，按 `--max-samples` 截断。
7. 如果指定 `--make-reverse`：
   - 用 `ffmpeg -vf reverse -an` 生成 `reverse` 片段
   - 输出到 `--reverse-dir/{clip_key}_rev.mp4`
8. 如果指定 `--make-composite`：
   - 先确保存在 `reverse` 片段
   - 如果没给 `--black-video`，先生成一段黑屏视频
   - 用 `ffmpeg concat` 拼出 `forward + black + reverse`
   - 输出到 `--composite-dir/{clip_key}_t2v.mp4`
9. 最终写出 manifest，每条记录至少包含：
   - `forward_video_path`
   - `reverse_video_path`
   - `composite_video_path`
   - `black_gap_sec`

补充说明：

- `forward_video_path` 指向原始 event clip，不会重新拷贝或重新导出一份 forward 视频。
- `reverse_video_path` 只有在 `--make-reverse` 或 `--make-composite` 触发生成时才会落盘。
- `composite_video_path` 只有在 `--make-composite` 时才会落盘。

### 2. `annotate_event_captions.py`

职责：为 manifest 中的 `forward` / `reverse` 视频生成文本描述。

实际逻辑：

1. 读取 manifest JSONL。
2. 对每条记录：
   - 取 `forward_video_path`
   - 取 `reverse_video_path`，如果为空，则退回到 `forward_video_path`
3. 用 `decord` 对每个视频均匀抽帧：
   - 总帧数小于 `--max-frames` 时全取
   - 否则按 stride 均匀采样
4. 将帧转成 JPEG data URL。
5. 用 OpenAI-compatible 接口分别请求两次 VLM：
   - 一次给 forward 视频
   - 一次给 reverse 视频
6. 期望模型返回 JSON：
   - `caption`
   - `confidence`
7. 汇总输出三份文件：
   - `forward_captions.jsonl`
   - `reverse_captions.jsonl`
   - `caption_pairs.jsonl`

`caption_pairs.jsonl` 里额外会记录：

- `forward_caption`
- `reverse_caption`
- `forward_confidence`
- `reverse_confidence`
- `is_different = (forward_caption != reverse_caption)`

补充说明：

- 这个脚本本身不做去重，默认相信 manifest 已经是去重后的。
- 如果 manifest 里没有真实的 `reverse_video_path`，脚本会把 forward 视频再次当成 reverse 去标注，因此这时正反 caption 很可能相同。

### 3. `build_aot_mcq.py`

职责：把 manifest 和 caption pair 转成训练用的 MCQ JSONL。

实际逻辑：

1. 读取 manifest，并用 `clip_key` 建索引。
2. 读取 `caption_pairs.jsonl`，打乱顺序。
3. 对每个 pair 逐条过滤：
   - `forward_confidence >= --min-confidence`
   - `reverse_confidence >= --min-confidence`
   - `is_different == True`
   - `clip_key` 必须在 manifest 中存在
4. 构造 `aot_v2t`：
   - 对 `forward_video_path` 产出一条，正确答案是 `A`
   - 如果有 `reverse_video_path`，再对 reverse 产出一条，正确答案是 `B`
5. 构造 `aot_t2v`：
   - 只有存在 `composite_video_path` 时才生成
   - 用 `forward_caption` 询问时答案固定是 `A`
   - 用 `reverse_caption` 询问时答案固定是 `B`
6. 分别写出：
   - `v2t_train.jsonl`
   - `t2v_train.jsonl`

补充说明：

- 这个脚本不会再检查视频文件是否存在，默认依赖上游 manifest 的路径有效。
- `kept` 是按 pair 计数，不是按最终输出条数计数，因此一个 pair 可能扩成多条训练样本。

### 4. `prompts.py`

职责：集中维护 prompt。

包含三类模板：

1. `SYSTEM_PROMPT`
   约束模型关注可见动作和时序方向。
2. `get_forward_reverse_caption_prompt()`
   用于单段视频 caption 标注，要求输出 JSON。
3. `get_v2t_prompt()` / `get_t2v_prompt()`
   用于构造最终 MCQ 问题。

## 检查结论

### 是否对视频片段做了去重

做了，但范围是“按视频路径精确去重”。

- 去重位置：`build_event_aot_data.py`
- 去重键：`video_path`
- 实现方式：用 `seen` 集合过滤，重复路径只保留第一次出现

这意味着：

- 如果同一个 event clip 在输入 JSONL 中重复出现多次，会被去重。
- 如果两个文件内容相同但路径不同，当前不会识别为重复。

### 是否保存了 forward 和 reverse 两个片段

当前代码不是“同时离线保存两份素材”，而是：

- `forward`
  - 始终在 manifest 中记录为 `forward_video_path`
  - 指向原始 event clip
  - 不会额外复制一份到新目录
- `reverse`
  - 只有在 `--make-reverse` 或 `--make-composite` 时才会实际生成并保存
  - 生成路径为 `--reverse-dir/{clip_key}_rev.mp4`

所以更准确地说：

- forward 片段会被“引用并记录”
- reverse 片段会被“按需生成并保存”

如果你的要求是“把 forward 和 reverse 都统一落到新的输出目录里”，那当前脚本还没有实现 forward 的拷贝/导出逻辑。

## 0. 依赖

```bash
pip install openai pillow decord
```

另外需要：

- `ffmpeg`
- 一个 OpenAI-compatible VLM endpoint

## 1. 生成 event manifest + 倒放素材

只生成 manifest：

```bash
python proxy_data/temporal_aot/build_event_aot_data.py \
  --input-jsonl proxy_data/proxy_train_easyr1.jsonl \
  --output-jsonl /path/to/aot_event_manifest.jsonl \
  --max-samples 1000 \
  --min-duration 3
```

同时离线导出 reverse clip：

```bash
python proxy_data/temporal_aot/build_event_aot_data.py \
  --input-jsonl proxy_data/proxy_train_easyr1.jsonl \
  --output-jsonl /path/to/aot_event_manifest.jsonl \
  --reverse-dir /path/to/reverse_clips \
  --make-reverse \
  --max-samples 1000 \
  --min-duration 3
```

同时生成 T2V composite：

```bash
python proxy_data/temporal_aot/build_event_aot_data.py \
  --input-jsonl proxy_data/proxy_train_easyr1.jsonl \
  --output-jsonl /path/to/aot_event_manifest.jsonl \
  --reverse-dir /path/to/reverse_clips \
  --composite-dir /path/to/t2v_composite \
  --make-reverse \
  --make-composite \
  --black-gap-sec 2.0 \
  --max-samples 1000 \
  --min-duration 3
```

说明：

- `--max-samples` 控制最终参与 AoT 构造的 event clip 数量
- `--min-duration` 用于过滤过短事件
- 输入直接用 `proxy_train_easyr1.jsonl`，因为里面已经能拿到 event clip 路径
- 这一步已经完成 event clip 的路径级去重
- 如果只跑 manifest 而不加 `--make-reverse`，manifest 里的 `reverse_video_path` 会为空

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

输出目录里会生成：

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

第一轮建议这样跑：

1. 先用 `--max-samples 100 ~ 300` 做一小批 event clip
2. 先只检查 `caption_pairs.jsonl` 质量
3. 先构造 `V2T`
4. `T2V` 确认 composite 视频没问题后再加

## 5. 当前实现的注意点

1. `annotate_event_captions.py` 在缺少 `reverse_video_path` 时，会回退到 forward 视频继续标注。
2. `build_aot_mcq.py` 只有在 manifest 存在 `composite_video_path` 时才会产出 `aot_t2v`。
3. 去重只基于路径，不基于视频内容哈希。
4. 当前不会单独导出一份新的 forward 素材目录；如果后续需要统一管理正反样本，建议在 `build_event_aot_data.py` 里补一个 forward export/copy 选项。
