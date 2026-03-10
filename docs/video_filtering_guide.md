# 视频过滤脚本使用指南

## 功能概述

`proxy_data/filter_bad_videos.py` 是一个用于检查和过滤混合训练数据集中视频问题的工具。它检查以下几类问题：

### 1. **不可读视频**（会导致训练崩溃）
- 文件不存在
- 文件为空
- decord 无法读取
- 帧数为 0

### 2. **帧数不足**（会触发 FRAME_FACTOR 错误）
- Qwen3-VL 要求 `temporal_patch_size=2`，即视频至少需要 **2 帧**
- 脚本默认过滤掉帧数 < 4 的视频（留余量应对多视频场景）
- 可通过 `--min-frames` 参数调整阈值

### 3. **时长不匹配**（可读但标注与实际不符）
- YouCook2 视频文件名包含时间戳：`event00_42_51.mp4`（表示 42-51 秒的片段）
- 脚本检查文件名标注时长 vs 实际视频时长的差异
- 允许 ±10% 误差（帧率/采样问题）
- 超过此阈值的视频会被标记为"时长不匹配"，**不会被自动过滤**（仅警告）

---

## 使用方法

### 基础用法：仅检查，不输出

```bash
python3 proxy_data/filter_bad_videos.py -i proxy_data/mixed_train_cot.jsonl
```

输出示例：
```
📂 读取数据集: proxy_data/mixed_train_cot.jsonl
  总样本数   : 9446
  唯一视频数  : 15580

🔍 验证视频可读性（workers=8）...
  ❌ [     1/15580] /path/to/video.mp4
             原因: 文件不存在
  ✅ [   500/15580] 进度...
  ...

📊 结果汇总:
  可读视频   : 12345 / 15580
  不可读视频  : 3235 / 15580
  时长不匹配  : 42 个（可读但标注与实际时长差异 >10%）

⚠️  时长不匹配诊断（标注vs实际，允许±10%误差):
  video_name_event00_42_51.mp4
    标注时长: 9.00s  →  实际时长: 10.20s  (差 1.20s, 13.3%)
  ...
```

### 完整过滤：生成干净数据集

```bash
python3 proxy_data/filter_bad_videos.py \
    -i proxy_data/mixed_train_cot.jsonl \
    -o proxy_data/mixed_train_cot_clean.jsonl \
    --min-frames 4 \
    --workers 16 \
    --bad_list proxy_data/bad_videos.txt
```

**参数说明：**
- `-i, --input`：输入 JSONL 文件路径
- `-o, --output`：输出干净 JSONL 文件路径（省略则仅检查）
- `--min-frames`：最小帧数要求，默认 4（Qwen3-VL 至少需要 2）
- `--workers`：并行检查线程数，默认 16（可根据 CPU 核数调整）
- `--bad_list`：保存不可读视频列表（可选）
- `--video_key`：JSON 中视频路径字段名，默认 "videos"
- `--filter-duration-mismatch`：同时丢弃时长不匹配的样本（默认关闭，仅警告）

### 示例：同时过滤时长不匹配的视频

```bash
python3 proxy_data/filter_bad_videos.py \
    -i proxy_data/mixed_train_cot.jsonl \
    -o proxy_data/mixed_train_cot_clean.jsonl \
    --min-frames 4 \
    --filter-duration-mismatch \
    --workers 32 \
    --bad_list proxy_data/bad_videos.txt
```

### 示例：针对 temporal_seg 任务的激进过滤

如果要求更严格的帧数要求（比如多视频场景）：

```bash
python3 proxy_data/filter_bad_videos.py \
    -i proxy_data/mixed_train_cot.jsonl \
    -o proxy_data/mixed_train_cot_clean.jsonl \
    --min-frames 8 \
    --workers 32 \
    --bad_list proxy_data/bad_videos.txt
```

---

## 时长不匹配诊断

当脚本输出"时长不匹配"警告时，说明文件名中的时间戳与实际视频时长差异较大。

### 常见原因

1. **视频文件损坏或转码**
   - 某些帧丢失或转码参数变化
   - 建议重新下载或检查原始文件

2. **帧率 (fps) 变化**
   - `smart_nframes` 计算依赖于实际 fps
   - 如果 fps 与预期不符，采样帧数会不同

3. **时间戳标注错误**（罕见）
   - 数据标注时错误记录了时间戳

### 处理建议

- **小差异（< 20%）**：通常可以忽略，qwen_vl_utils 会自动采样
- **大差异（> 50%）**：建议加上 `--filter-duration-mismatch` 一并丢弃
- **批量处理**：可在过滤后单独处理这些视频

---

## 预期过滤效果

对于 YouCook2 数据集：

| 指标 | 预期 |
|------|------|
| 总样本数 | ~9446 |
| 唯一视频数 | ~15580 |
| 不可读 | 5-10%（多为文件不存在） |
| 帧数不足 | 1-3%（极短视频片段） |
| 时长不匹配 | 0.1-0.5%（转码/fps问题） |
| **保留比例** | **85-95%** |

---

## 后续处理

### 步骤 1：过滤数据集

```bash
python3 proxy_data/filter_bad_videos.py \
    -i proxy_data/mixed_train_cot.jsonl \
    -o proxy_data/mixed_train_cot_clean.jsonl \
    --min-frames 4 \
    --workers 32
```

### 步骤 2：更新启动脚本

编辑 `local_scripts/run_mixed_proxy_training.sh`：

```bash
# 修改前
TRAIN_FILE="proxy_data/mixed_train_cot.jsonl"

# 修改后
TRAIN_FILE="proxy_data/mixed_train_cot_clean.jsonl"
```

### 步骤 3：（可选）保存坏视频列表

```bash
python3 proxy_data/filter_bad_videos.py \
    -i proxy_data/mixed_train_cot.jsonl \
    -o proxy_data/mixed_train_cot_clean.jsonl \
    --bad_list proxy_data/bad_videos_youcook2.txt
```

然后可分析 `bad_videos_youcook2.txt` 中的文件来源，评估数据质量。

---

## 故障排除

### Q: 显示 "nframes should in interval [2, 1], but got 0"

**A:** 说明有视频帧数不足，需要过滤。确保：
1. 使用了干净数据集（混合训练脚本中 `TRAIN_FILE` 指向 `*_clean.jsonl`）
2. `--min-frames` 值足够大，建议 ≥ 4

### Q: 大量视频显示"文件不存在"

**A:** 说明数据集中的视频路径不正确或数据集未正确挂载。
- 检查服务器上文件系统挂载：`mount | grep m2v`
- 检查路径前缀是否正确
- 可在服务器上运行脚本，而不是本地

### Q: 过滤后样本数少很多

**A:** 这是正常的。YouCook2 包含多视频样本（context + options = 6 个视频），只要有 1 个视频不可读，整个样本就被丢弃。
- 预期丢弃率：5-15%
- 如需保留更多样本，可降低 `--min-frames` 阈值（但需谨慎）

---

## 性能参考

| 参数 | 时间 |
|------|------|
| 读取 9446 样本、15580 唯一视频 | ~2s |
| 并行检查（workers=32） | ~30-60s（取决于 I/O） |
| 输出干净数据集 | ~5s |
| **总耗时** | **~1 分钟** |

---

## 相关文档

- [混合训练框架](./multi_video_task_batching_grpo.md)
- [KL 散度诊断](./kl_divergence_diagnostic.md)
