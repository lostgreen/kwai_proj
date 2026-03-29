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
- [ ] task × source 分布分析
- [ ] 文本筛选规则确定 & 执行
- [ ] 视频可用性验证（解压 + decord 检查）
- [ ] 领域均衡采样
- [ ] 人工审查

## 运行指令

### Step 1: 文本筛选

```bash
# 在服务器上执行，数据路径根据实际修改
cd proxy_data/data_curation/sources/et_instruct_164k

# Dry run（仅看统计，不写文件）
python text_filter.py \
    --json_path /m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/et_instruct_164k_txt.json \
    --config ../../configs/et_instruct_164k.yaml \
    --dry_run

# 正式筛选
python text_filter.py \
    --json_path /m2v_intern/xuboshen/zgw/data/ET-Instruct-164K/et_instruct_164k_txt.json \
    --output_dir results \
    --config ../../configs/et_instruct_164k.yaml
```

**产出**：
- `results/passed.jsonl` — 通过筛选的样本（含 `_origin` 溯源元数据）
- `results/rejected.jsonl` — 被拒样本
- `results/filter_summary.json` — 筛选统计摘要

### Step 2: LLM 层次潜力评估

```bash
# 抽样评估（默认 200 条）
python assess_hierarchy.py \
    --input results/passed.jsonl \
    --output results/assessed.jsonl \
    --sample-n 200 \
    --api-base https://api.novita.ai/v3/openai \
    --model pa/gmn-2.5-pr

# 全量评估（断点续评）
python assess_hierarchy.py \
    --input results/passed.jsonl \
    --output results/assessed.jsonl \
    --no-sample --resume \
    --workers 16
```

### Step 3: 可视化验证

```bash
# 回到项目根目录
cd ../../../../

# 启动可视化（candidate 模式）
python data_visualization/server.py \
    --candidate-data proxy_data/data_curation/sources/et_instruct_164k/results/passed.jsonl

# 浏览器打开 http://127.0.0.1:8787/#candidate
```
