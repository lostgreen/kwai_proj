# Data Visualization

统一可视化工具，支持三类数据的交互式查看：

| 标签页 | 数据类型 | 核心功能 |
|--------|---------|---------|
| **Segmentation** | annotation JSON / seg JSONL | 三层时序标注段条 + 1fps 帧条 + diagnostics |
| **AoT Caption** | `caption_pairs.jsonl` + manifest | forward / reverse / shuffle 帧条 + VLM caption 并排对比 |
| **AoT MCQ** | `v2t_train.jsonl` / `t2v_train.jsonl` | 视频帧 + A/B(/C/D) 选项卡 + 正确答案高亮 |

---

## 快速启动

```bash
cd data_visualization
./run.sh              # 空界面，在浏览器中手动填写路径
```

启动后访问 **http://127.0.0.1:8787/**

---

## Segmentation 标注可视化

### 方式 1 — 浏览器手动加载
```bash
./run.sh
# 在「Segmentation」标签页输入路径后点击「加载数据」
```

### 方式 2 — 命令行预加载（annotation JSON 目录）
```bash
./run.sh --data-path proxy_data/youcook2_seg_annotation/datasets/
```

### 方式 3 — 预加载单个 dataset JSONL（任一层级）
```bash
# L1 macro phase 标注
./run.sh --data-path proxy_data/youcook2_seg_annotation/datasets/youcook2_hier_L1_train_clipped.jsonl

# L2 cooking event 标注
./run.sh --data-path proxy_data/youcook2_seg_annotation/datasets/youcook2_hier_L2_train_clipped.jsonl

# L3 atomic grounding 标注
./run.sh --data-path proxy_data/youcook2_seg_annotation/datasets/youcook2_hier_L3_train_clipped.jsonl

# 混合三层训练集
./run.sh --data-path proxy_data/youcook2_seg_annotation/datasets/youcook2_hier_mixed_train.jsonl
```

### 方式 4 — 限制预加载数量（大数据集提速）
```bash
./run.sh --data-path proxy_data/youcook2_seg_annotation/datasets/youcook2_hier_mixed_train.jsonl \
         --max-samples 200 --prefer-complete
```

---

## AoT Caption 可视化

查看 forward / reverse / shuffle 三方向视频帧和 VLM 生成的 caption 并排对比。

### 仅加载 caption pairs（无视频帧预览）
```bash
./run.sh --caption-pairs proxy_data/temporal_aot/data/aot_annotations/caption_pairs.jsonl
```

### 同时加载 manifest（显示视频帧条）
```bash
./run.sh --caption-pairs proxy_data/temporal_aot/data/aot_annotations/caption_pairs.jsonl \
         --manifest     proxy_data/temporal_aot/data/aot_event_manifest.jsonl
```

### 加载完整 manifest（包含所有 clips）
```bash
./run.sh --caption-pairs proxy_data/temporal_aot/data/aot_annotations/caption_pairs.jsonl \
         --manifest     proxy_data/temporal_aot/data/aot_event_manifest_all.jsonl
```

---

## AoT MCQ 可视化

查看 Video-to-Text (V2T) 和 Text-to-Video (T2V) 单选题，包括视频帧条、选项卡片和正确答案高亮。

### V2T（视频 → 选择 caption）
```bash
./run.sh --mcq-data proxy_data/temporal_aot/data/aot_annotations/v2t_train.jsonl
```

### T2V（caption → 选择视频）
```bash
./run.sh --mcq-data proxy_data/temporal_aot/data/aot_annotations/t2v_train.jsonl
```

### 混合训练集（包含多种题型）
```bash
./run.sh --mcq-data proxy_data/temporal_aot/data/mixed_aot_train.jsonl
```

### 过滤后的混合集
```bash
./run.sh --mcq-data proxy_data/temporal_aot/data/mixed_aot_train.offline_filtered.jsonl
```

---

## 同时预加载多类数据

所有预加载参数可以组合使用，三个标签页各自独立显示：

```bash
./run.sh \
  --data-path     proxy_data/youcook2_seg_annotation/datasets/youcook2_hier_mixed_train.jsonl \
  --caption-pairs proxy_data/temporal_aot/data/aot_annotations/caption_pairs.jsonl \
  --manifest      proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
  --mcq-data      proxy_data/temporal_aot/data/aot_annotations/v2t_train.jsonl
```

---

## 其他选项

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--port` | 监听端口 | `8787` |
| `--host` | 监听地址 | `127.0.0.1` |
| `--max-samples N` | seg 预加载最多 N 条（大数据集限速） | 不限 |
| `--prefer-complete` | 优先预加载有完整三层标注的 clip | 关闭 |

### 示例：自定义端口
```bash
./run.sh --port 9090 --data-path proxy_data/youcook2_seg_annotation/datasets/
```

### 直接用 Python 启动
```bash
python data_visualization/server.py --port 8787 --mcq-data /absolute/path/to/mcq.jsonl
```

---

## 文件说明

```
data_visualization/
├── server.py   — 统一后端：SegmentationStore + AoTCaptionStore + AoTMCQStore
├── index.html  — 统一前端：三标签页，帧条 / 段条 / 选项卡 复用同一套 CSS 组件
├── run.sh      — 快捷启动脚本（透传所有参数给 server.py）
├── DESIGN.md   — 完整架构设计文档
└── segmentation_visualize/  ← 旧版独立工具，保留向后兼容
```
