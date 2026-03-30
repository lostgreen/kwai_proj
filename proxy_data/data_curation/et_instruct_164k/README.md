# ET-Instruct-164K 数据源

## 基本信息

- **来源**: ET-Instruct-164K
- **路径**: `/m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/`
- **规模**: ~164K 样本
- **文件**:
  - `et_instruct_164k_vid.json` — 视频相关样本
  - `et_instruct_164k_txt.json` — 纯文本样本
- **视频域**: ActivityNet, COIN, DiDeMo, Ego4D-NAQ, Ego-TimeQA, HACS, HowToCaption, HowToStep, MR-HiSum, QuerYD, TACoS, ViTT

## 数据格式

来源论文: [E.T. Bench (NeurIPS 2024)](https://arxiv.org/abs/2409.18111)

使用 `et_instruct_164k_txt.json`（纯文本时间戳，非 `<vid>` token 版本）。

```json
{
  "task": "slc",                           // 任务类型（9 种）
  "source": "how_to_step",                 // 来源数据集 = domain
  "video": "how_to_step/PJi8ZEHAFcI.mp4",  // 视频相对路径
  "duration": 200.767,                     // 视频时长（秒）
  "src": [12, 18],                         // [可选] 输入时间戳（秒）
  "tgt": [36, 44, 49, 57],                 // [可选] 输出时间戳（秒），成对出现 [start, end, ...]
  "conversations": [                       // 对话对
    {"from": "human", "value": "<image>\n..."},
    {"from": "gpt", "value": "36.0 - 44.0 seconds, clean the bananas. ..."}
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `task` | str | 任务类型：slc, rvq, grounding, dense_cap 等（9 种） |
| `source` | str | 来源域：activitynet, coin, didemo, ego4d_naq, ego_timeqa, hacs, how_to_caption, how_to_step, mr_hisum, queryd, tacos, vitt |
| `video` | str | 相对视频路径 `{source}/{id}.mp4` |
| `duration` | float | 视频时长（秒），平均约 146s |
| `tgt` | list[int] | 输出时间戳列表，成对 [s1,e1,s2,e2,...] |
| `src` | list[int] | 输入时间戳（可选） |

### 视频规格
- 已处理为 3 FPS, 224px shortest side, 无音频

### Domains & 视频包大小

| Domain | tar.gz 大小 |
|--------|------------|
| how_to_step | 45G |
| how_to_caption | 40G |
| hacs | 39G |
| ego_timeqa | 34G |
| mr_hisum | 25G |
| activitynet | 20G |
| coin | 14G |
| didemo | 8.1G |
| vitt | 7.4G |
| ego4d_naq | 2.9G |
| queryd | 1.3G |
| tacos | 24M |

## 筛选进度

- [x] 数据格式探索
- [x] 文本筛选规则确定 & 执行
- [ ] Stage A: L2 粒度粗筛
- [ ] Stage B: 层次潜力精筛
- [ ] 可视化验证
- [ ] 领域均衡采样

## 运行指令

> **所有命令均从 `train/` (EasyR1) 目录执行**，使用相对路径。

### Step 0: 文本筛选

```bash
# Dry run（仅看统计，不写文件）
python proxy_data/data_curation/et_instruct_164k/text_filter.py \
    --json_path /m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/et_instruct_164k_txt.json \
    --config proxy_data/data_curation/configs/et_instruct_164k.yaml \
    --dry_run

# 正式筛选
python proxy_data/data_curation/et_instruct_164k/text_filter.py \
    --json_path /m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/et_instruct_164k_txt.json \
    --output_dir proxy_data/data_curation/et_instruct_164k/results \
    --config proxy_data/data_curation/configs/et_instruct_164k.yaml
```

**产出**：
- `proxy_data/data_curation/results/et_instruct_164k/passed.jsonl`
- `proxy_data/data_curation/results/et_instruct_164k/rejected.jsonl`
- `proxy_data/data_curation/results/et_instruct_164k/filter_summary.json`

### Stage A: L2 粒度粗筛

```bash
# 抽样 200 条看分布
python proxy_data/data_curation/et_instruct_164k/stage_a_coarse_filter.py \
    --input proxy_data/data_curation/results/et_instruct_164k/passed.jsonl \
    --output proxy_data/data_curation/results/et_instruct_164k/stage_a_results.jsonl \
    --sample-n 200

# 全量评估（断点续评）
python proxy_data/data_curation/et_instruct_164k/stage_a_coarse_filter.py \
    --input proxy_data/data_curation/results/et_instruct_164k/passed.jsonl \
    --output proxy_data/data_curation/results/et_instruct_164k/stage_a_results.jsonl \
    --no-sample --resume --workers 16

# 查看 Stage A 分析报告
python proxy_data/data_curation/shared/analyze_results.py \
    --input proxy_data/data_curation/results/et_instruct_164k/stage_a_results.jsonl \
    --stage A --review 3 \
    --html proxy_data/data_curation/results/et_instruct_164k/stage_a_report.html
```

### Stage B: 层次潜力精筛

```bash
python proxy_data/data_curation/shared/vision_filter.py \
    --input proxy_data/data_curation/results/et_instruct_164k/stage_a_results_keep.jsonl \
    --output proxy_data/data_curation/results/et_instruct_164k/stage_b_results.jsonl \
    --data-source et_instruct --no-sample --resume --workers 16

# 查看 Stage B 分析报告
python proxy_data/data_curation/shared/analyze_results.py \
    --input proxy_data/data_curation/results/et_instruct_164k/stage_b_results.jsonl \
    --stage B --review 3 \
    --html proxy_data/data_curation/results/et_instruct_164k/stage_b_report.html
```

### 可视化验证（帧 + 时间线）

```bash
# 转换 keep 样本为 segmentation_visualize 格式（含 1fps 抽帧）
# --output 建议指向数据集目录，避免大量帧文件进入 git
python proxy_data/data_curation/shared/convert_to_viz.py \
    --input proxy_data/data_curation/results/et_instruct_164k/stage_a_results_keep.jsonl \
    --output /m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/viz_candidates/ \
    --data-source et_instruct \
    --video-root /m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/videos/ \
    --workers 8

# 仅生成 JSON（不抽帧）
# python proxy_data/data_curation/shared/convert_to_viz.py \
#     --input ... --output ... --data-source et_instruct --no-frames

# 启动可视化服务器
python data_visualization/segmentation_visualize/server.py \
    --annotation-dir /m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/viz_candidates/ \
    --port 8765

# 浏览器打开 http://127.0.0.1:8765
```
