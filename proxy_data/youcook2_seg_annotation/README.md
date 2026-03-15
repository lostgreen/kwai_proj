# YouCook2 Hierarchical DVC Annotation Pipeline

3-level hierarchical Dense Video Captioning (DVC) annotation tool for YouCook2 windowed clips.

## Directory layout

```
youcook2_seg_annotation/
  prompts.py          — Annotation prompts (Level 0-3; Levels 2 & 3 are TODO placeholders)
  extract_frames.py   — ffmpeg 1fps frame extraction
  annotate.py         — LLM annotation pipeline (OpenAI-compatible VLM API)
  build_dataset.py    — Convert annotations → EasyR1 training JSONL
  README.md
  frames/             — extracted frames (git-ignored)
  annotations/        — per-clip annotation JSONs (git-ignored)
```

## Annotation levels

| Level | Name | Status | Prompt |
|---|---|---|---|
| 0 | System Prompt | ✅ Provided | `prompts.SYSTEM_PROMPT` |
| 1 | Macro Phase (阶段级) | ✅ Active | `prompts.get_level1_prompt()` |
| 2 | Activity-level (活动级) | 🔲 TODO | `prompts._LEVEL2_BASE` — fill in when ready |
| 3 | Atomic Step (动作级) | 🔲 TODO | `prompts._LEVEL3_BASE` — fill in when ready |

## Quick start

### Step 1: Extract 1fps frames

```bash
python proxy_data/youcook2_seg_annotation/extract_frames.py \
  --jsonl proxy_data/youcook2_train_easyr1.jsonl \
  --video-dir /path/to/Youcook2_windowed \
  --output-dir proxy_data/youcook2_seg_annotation/frames \
  --fps 1.0 \
  --workers 8
```

### Step 2: Run Level 1 annotation

Start a Qwen3-VL server (e.g. vLLM), then:

```bash
python proxy_data/youcook2_seg_annotation/annotate.py \
  --jsonl proxy_data/youcook2_train_easyr1.jsonl \
  --frames-dir proxy_data/youcook2_seg_annotation/frames \
  --output-dir proxy_data/youcook2_seg_annotation/annotations \
  --level 1 \
  --api-base http://localhost:8000/v1 \
  --model Qwen3-VL-7B \
  --workers 4 \
  --max-frames-per-call 32

  python /home/xuboshen/zgw/EasyR1/proxy_data/youcook2_seg_annotation/annotate.py \
        --frames-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/frames \
        --output-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations \
        --level 1 \
        --api-base https://api.novita.ai/v3/openai \
        --model pa/gmn-2.5-pr \
        --workers 4 \
        --limit 50 \
        --max-frames-per-call 1024

  python /home/xuboshen/zgw/EasyR1/proxy_data/youcook2_seg_annotation/annotate.py \
    --frames-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/frames \
    --output-dir /m2v_intern/xuboshen/zgw/data/youcook2_seg_annotation/annotations \
    --level 2 \
    --api-base https://api.novita.ai/v3/openai \
    --model pa/gmn-2.5-pr \
    --workers 4 \
    --limit 5 \
    --max-frames-per-call 1024 \
    --overwrite

```

### Step 3: Build training dataset

```bash
python proxy_data/youcook2_seg_annotation/build_dataset.py \
  --annotation-dir proxy_data/youcook2_seg_annotation/annotations \
  --output proxy_data/youcook2_seg_annotation/youcook2_hier_L1_train.jsonl \
  --level 1
```

## Level 2 & 3: adding prompts

Edit `prompts.py`, replace `_LEVEL2_BASE` and `_LEVEL3_BASE` with the actual prompt text.
The `get_level2_prompt()` / `get_level3_prompt()` functions will automatically inject
the clip duration and previous-level context, then raise `NotImplementedError` until
the base strings are filled in.

## Annotation JSON format

```json
{
  "clip_key": "GLd3aX16zBg_90_174",
  "video_path": "/m2v_intern/.../GLd3aX16zBg_90_174.mp4",
  "clip_duration_sec": 84.0,
  "n_frames": 84,
  "frame_dir": "frames/GLd3aX16zBg_90_174",
  "level1": {
    "macro_phases": [
      {"phase_id": 1, "start_time": "00:00", "end_time": "00:18",
       "phase_name": "Ingredient Preparation",
       "narrative_summary": "Spreads margarine on bread slices."}
    ]
  },
  "level2": null,
  "level3": null,
  "annotated_at": "2025-..."
}
```
