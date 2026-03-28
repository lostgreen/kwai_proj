# Prompt Ablation Comparison Studio

Compare segmentation rollouts from 4 prompt variants (V1–V4) side-by-side.

## Prompt Variants (2×2 Design)

|           | No-CoT      | Structured CoT |
|-----------|-------------|----------------|
| Minimal   | **V1** Base | **V3** CoT     |
| Gran-Enh  | **V2** Gran | **V4** Gran+CoT|

## Quick Start

```bash
# Default paths (edit run.sh or override via env vars)
bash prompt_ablation_visualization/run.sh

# Custom paths
MODEL_ROOT=/your/model/root \
PORT=8891 \
bash prompt_ablation_visualization/run.sh

# Fully custom
EXP_DIRS="V1=/path/V1/rollouts,V2=/path/V2/rollouts,V3=/path/V3/rollouts,V4=/path/V4/rollouts" \
LOG_FILES="V1=/path/V1/log.jsonl,V2=/path/V2/log.jsonl" \
bash prompt_ablation_visualization/run.sh
```

Open `http://localhost:8891` in browser.

## Features

### Overview Page
- Per-variant summary cards (groups, samples, mean reward)
- Reward curves overlay (4 lines, filterable by task level)
- Per-level bar chart comparison (L1/L2/L3 final rewards)
- Training metrics comparison (KL, PG loss, grad norm, etc.)

### Sample Browser
- Step selector with reward summary per variant
- Sample list sorted by reward spread (most interesting first)
- **2×2 Timeline Comparison**: GT/Pred segments for each variant
- Shared video frame strip (loaded once, displayed once)
- Response text comparison with tab switching
- CoT highlighting (`<think>` / `<events>` tags)

### Prompt Inspector
- Side-by-side prompt template comparison for each variant
- See exactly how V1→V4 prompts differ

## Architecture

```
prompt_ablation_visualization/
├── server.py      # Backend: multi-experiment JSONL loader + API
├── index.html     # Frontend: SPA with 3 pages
├── run.sh         # Launch script
└── README.md
```

Backend reuses the same rollout JSONL format and frame extraction
logic from `rollout_visualization/`, adapted for 4 parallel experiments.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/overview` | Variant summaries |
| `GET /api/steps` | Common step list |
| `GET /api/samples?step_key=...&task=...` | Aligned samples |
| `GET /api/sample/{uid}` | Full comparison for one sample |
| `GET /api/frames/{uid}` | Shared frames (lazy load) |
| `GET /api/reward_curves` | All reward curves |
