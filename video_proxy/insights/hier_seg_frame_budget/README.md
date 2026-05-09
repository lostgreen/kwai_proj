# Hier-Seg Frame Budget Insights

Analyze hierarchical segmentation annotation durations and estimate how much
ground-truth resolution remains under frame budgets such as `48`, `64`, `128`,
and `256`.

## Run

```bash
python video_proxy/insights/hier_seg_frame_budget/analyze.py \
  --annotation-dir /path/to/youcook2_seg/hier_seg_annotation/annotations \
  --output-dir video_proxy/insights/hier_seg_frame_budget/outputs
```

Outputs include summary CSV/JSON files and PNG charts.
