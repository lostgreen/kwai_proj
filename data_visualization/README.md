# Data Visualization

统一可视化工具，支持三类数据的交互式查看。所有数据路径均通过**启动参数或环境变量**传入，服务端预加载后页面打开即可浏览，无需在浏览器中输入路径。

| 标签页 | 数据 | 核心功能 |
|--------|------|---------|
| **Segmentation** | annotation JSON / seg JSONL | 三层时序标注段条 + 1fps 帧条 + diagnostics |
| **AoT Caption** | `caption_pairs.jsonl` + `aot_event_manifest.jsonl` | forward / reverse / shuffle 帧条 + VLM caption 并排对比 |
| **AoT MCQ** | `v2t / t2v / 4way` JSONL | 视频帧 + A/B(/C/D) 选项卡 + 正确答案高亮 |

启动后访问 **http://127.0.0.1:8890/**

---

## Segmentation 标注可视化

```bash
# 加载 annotation JSON 目录
./data_visualization/run.sh --data-path proxy_data/hier_seg_annotation/datasets/

# 加载单个 dataset JSONL（任一层级）
./data_visualization/run.sh --data-path proxy_data/hier_seg_annotation/datasets/youcook2_hier_mixed_train.jsonl

# 限制预加载数量（大数据集提速）
./data_visualization/run.sh --data-path proxy_data/hier_seg_annotation/datasets/youcook2_hier_mixed_train.jsonl \
  --max-samples 200 --prefer-complete

# 远端服务器（env-var 方式，路径较长时更方便）
DATA_PATH=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1/annotations \
  ./data_visualization/run.sh
```

---

## AoT Caption 可视化

**必须同时提供 `--manifest`，否则视频帧无法抽取。**

```bash
# 标准启动（有视频帧预览）
./data_visualization/run.sh \
  --caption-pairs proxy_data/temporal_aot/data/aot_annotations/caption_pairs.jsonl \
  --manifest      proxy_data/temporal_aot/data/aot_event_manifest.jsonl

# 远端服务器（env-var 方式）
CAPTION_PAIRS=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/caption_pairs.jsonl \
MANIFEST=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/aot_event_manifest.jsonl \
  ./data_visualization/run.sh
```

---

## AoT MCQ 可视化

```bash
# V2T（视频 → 选 caption）
./data_visualization/run.sh --mcq-data proxy_data/temporal_aot/data/aot_annotations/v2t_train.jsonl

# T2V（caption → 选视频）
./data_visualization/run.sh --mcq-data proxy_data/temporal_aot/data/aot_annotations/t2v_train.jsonl

# 混合训练集
./data_visualization/run.sh --mcq-data proxy_data/temporal_aot/data/mixed_aot_train.jsonl

# 远端服务器（env-var 方式）
MCQ_DATA=/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/youcook2_aot/v2t_train.jsonl \
  ./data_visualization/run.sh
```

---

## 同时预加载多类数据

所有参数可以自由组合，三个标签页各自独立显示：

```bash
./data_visualization/run.sh \
  --data-path     proxy_data/hier_seg_annotation/datasets/youcook2_hier_mixed_train.jsonl \
  --caption-pairs proxy_data/temporal_aot/data/aot_annotations/caption_pairs.jsonl \
  --manifest      proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --mcq-data      proxy_data/temporal_aot/data/aot_annotations/v2t_train.jsonl
```

---

## 参数说明

| 参数 / 环境变量 | 说明 | 默认值 |
|----------------|------|--------|
| `--data-path` / `DATA_PATH` | seg annotation 目录或 JSONL | — |
| `--caption-pairs` / `CAPTION_PAIRS` | caption_pairs.jsonl 路径 | — |
| `--manifest` / `MANIFEST` | aot_event_manifest.jsonl（配合 caption 使视频帧可见） | — |
| `--mcq-data` / `MCQ_DATA` | MCQ JSONL 路径 | — |
| `--port` / `PORT` | 监听端口 | `8890` |
| `--host` | 监听地址 | `127.0.0.1` |
| `--max-samples N` / `MAX_SAMPLES` | seg 预加载最多 N 条 | 不限 |
| `--prefer-complete` | 优先加载有完整三层标注的 clip | 关闭 |

---

## 文件说明

```
data_visualization/
├── server.py   — 统一后端：SegmentationStore + AoTCaptionStore + AoTMCQStore
├── index.html  — 统一前端：三标签页，帧条 / 段条 / 选项卡复用同一套 CSS 组件
├── run.sh      — 启动脚本，支持 CLI 参数和环境变量两种方式
├── DESIGN.md   — 完整架构设计文档
└── segmentation_visualize/  ← 旧版独立工具，保留向后兼容
```

