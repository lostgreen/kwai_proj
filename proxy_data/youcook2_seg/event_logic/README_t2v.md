# Event Logic Proxy Task — T→V Extension

This document describes the **T→V (text-context → video-option)** extension to the Event Logic proxy task, which complements the existing **V→T (video-context → text-option)** tasks.

## Overview

The two task directions are modality-symmetric:

| Direction | Context | Options | Tests |
|---|---|---|---|
| **V→T** (existing) | N video clips showing cooking steps | 4 text descriptions | Given what you *see*, predict the next/missing step in words |
| **T→V** (new) | N recipe step descriptions (text) | 4 video clips | Given what you *read*, identify the next/missing step as a video |

Both use the same YouCook2 event clip data. The ablation is: **does the modality of the task context change what the model learns?**

---

## Data Sources & File Map

```
proxy_data/
├── temporal_aot/data/
│   ├── aot_event_manifest.jsonl         ← AoT events (1000 clips, each per line)
│   └── aot_annotations/
│       └── forward_captions.jsonl       ← AoT-style captions (temporal direction)
│
└── event_logic/
    ├── data/
    │   ├── proxy_train_text_options.jsonl    ← V→T training data (add/replace, text options)
    │   ├── l2_step_captions.jsonl            ← [NEW] Recipe-instruction captions per clip
    │   └── l2_event_logic_t2v.jsonl          ← [NEW] T→V training data (add_t2v/replace_t2v)
    │
    ├── annotate_l2_step_captions.py     ← [NEW] Generate recipe-instruction captions
    ├── build_l2_event_logic_t2v.py      ← [NEW] Build T→V dataset
    ├── build_l2_event_logic.py          ← V→T + Sort dataset builder (existing)
    └── prompts.py                       ← All prompt templates (updated)
```

### Caption style comparison

| Source | Format | Designed for |
|---|---|---|
| `aot_annotations/forward_captions.jsonl` | "First A, then B, finally C" (temporal direction markers) | AoT task: distinguish forward vs reverse direction |
| `l2_step_captions.jsonl` (**new**) | "Add the diced onions and sauté until translucent." (recipe instruction) | T→V: readable step-by-step context for causal reasoning |

---

## Full Pipeline

### Step 0 — Prerequisites

```bash
# These should already exist:
# proxy_data/event_logic/data/proxy_train_text_options.jsonl  (V→T data, already built)
# proxy_data/temporal_aot/data/aot_event_manifest.jsonl       (1000 AoT events)

# Environment
export NOVITA_API_KEY="your-key-here"
export API_BASE="https://api.novita.ai/v3/openai"
export MODEL="qwen/qwen2.5-vl-72b-instruct"
```

### Step 1 — Generate recipe-instruction captions

**Input**: `data/proxy_train_text_options.jsonl` — existing V→T dataset (has all event clip paths)
**Output**: `data/l2_step_captions.jsonl` — one caption per unique event clip

```bash
python proxy_data/event_logic/annotate_l2_step_captions.py \
    --from-dataset  proxy_data/event_logic/data/proxy_train_text_options.jsonl \
    --output        proxy_data/event_logic/data/l2_step_captions.jsonl \
    --api-base      "$API_BASE" \
    --model         "$MODEL" \
    --workers       8 \
    --max-frames    16 \
    --confidence-threshold 0.7
```

**Expected output format** (`l2_step_captions.jsonl`):
```json
{"clip_key": "WlHWRPyA7_g_event04_95_112", "video_path": "...", "caption": "Add the wonton filling to the center of the wrapper and fold the edges to seal tightly.", "confidence": 0.92}
{"clip_key": "WlHWRPyA7_g_event05_123_131", "video_path": "...", "caption": "Place the assembled wontons onto a floured tray to prevent sticking.", "confidence": 0.88}
```

**Coverage check**:
```bash
wc -l proxy_data/event_logic/data/l2_step_captions.jsonl
# Should be ≈ unique clips in proxy_train_text_options.jsonl
python -c "
import json; from pathlib import Path
clips = set()
for l in open('proxy_data/event_logic/data/proxy_train_text_options.jsonl'):
    for v in json.loads(l).get('videos', []): clips.add(Path(v).stem)
caps = {json.loads(l)['clip_key'] for l in open('proxy_data/event_logic/data/l2_step_captions.jsonl')}
print(f'Clips needing caption: {len(clips)}')
print(f'Clips with caption: {len(clips & caps)}')
print(f'Missing: {len(clips - caps)}')
"
```

### Step 2 — Build T→V dataset

**Input**:
- `aot_event_manifest.jsonl` — event sequence metadata (source_video_id, sequence_index, clip_key, forward_video_path)
- `l2_step_captions.jsonl` — captions from Step 1

**Output**: `data/l2_event_logic_t2v.jsonl` — T→V training data

```bash
# Without AI filter (faster, for quick iteration)
python proxy_data/event_logic/build_l2_event_logic_t2v.py \
    --manifest-jsonl proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
    --captions-jsonl proxy_data/event_logic/data/l2_step_captions.jsonl \
    --output         proxy_data/event_logic/data/l2_event_logic_t2v.jsonl \
    --seed 42 --shuffle

# With AI causality filter (recommended for training)
python proxy_data/event_logic/build_l2_event_logic_t2v.py \
    --manifest-jsonl proxy_data/temporal_aot/data/aot_event_manifest.jsonl \
    --captions-jsonl proxy_data/event_logic/data/l2_step_captions.jsonl \
    --output         proxy_data/event_logic/data/l2_event_logic_t2v_filtered.jsonl \
    --filter \
    --api-base       "$API_BASE" \
    --model          "$MODEL" \
    --confidence-threshold 0.75 \
    --filter-workers 8 \
    --seed 42 --shuffle
```

**Expected output format** (`l2_event_logic_t2v.jsonl`):
```json
{
  "messages": [{"role": "user", "content": "Context Sequence (recipe steps):\nStep 1: Chop the onions...\nStep 2: Heat the pan...\n\nWhich video clip shows the next step?\nOptions:\nA. <video>\nB. <video>\nC. <video>\nD. <video>\n\nThink step by step inside <think></think>..."}],
  "answer": "C",
  "videos": ["/path/event_A.mp4", "/path/event_B.mp4", "/path/event_C.mp4", "/path/event_D.mp4"],
  "data_type": "video",
  "problem_type": "add_t2v",
  "metadata": {"source_video_id": "WlHWRPyA7_g", "target_clip_key": "...", ...}
}
```

---

## Smoke Test

Run these checks before launching full training:

```bash
cd proxy_data/event_logic

# 1. Verify caption script CLI
python annotate_l2_step_captions.py --help

# 2. Verify T→V build script CLI
python build_l2_event_logic_t2v.py --help

# 3. Dry-run caption annotation on 5 clips (no API call, just check manifest extraction)
python -c "
from annotate_l2_step_captions import build_manifest_from_dataset
recs = build_manifest_from_dataset('data/proxy_train_text_options.jsonl')
print(f'Manifest: {len(recs)} unique clips')
print('First 3:')
for r in recs[:3]: print(' ', r['clip_key'], '|', r['video_path'][:60])
"

# 4. Dry-run T→V build (no API filter) using dummy captions
python -c "
import json, random
from pathlib import Path
from build_l2_event_logic_t2v import load_manifest, build_video_sequences, build_add_t2v_sample, build_replace_t2v_sample

records = load_manifest('../temporal_aot/data/aot_event_manifest.jsonl')
# Use original 'sentence' as fake captions for dry run
captions = {r['clip_key']: r['sentence'] for r in records if r.get('sentence')}
seqs, pool = build_video_sequences(records, captions, min_events=4)
print(f'Sequences: {len(seqs)}, Pool: {len(pool)}')

rng = random.Random(42)
add_s = build_add_t2v_sample(seqs[0], pool, rng=rng)
rep_s = build_replace_t2v_sample(seqs[0], pool, rng=rng)
print('add_t2v sample:')
print('  answer:', add_s['answer'])
print('  num_videos:', len(add_s['videos']))
print('  problem_type:', add_s['problem_type'])
print('replace_t2v sample:')
print('  answer:', rep_s['answer'])
print('  num_videos:', len(rep_s['videos']))
print('  problem_type:', rep_s['problem_type'])
print('[SMOKE TEST PASSED]')
"

# 5. Check reward function supports new problem types (after updating mixed_proxy_reward.py)
# python -c "
# from verl.reward_function.mixed_proxy_reward import compute_reward
# print('add_t2v' in str(compute_reward.__code__.co_consts))  # should be True after update
# "
```

---

## Ablation Design

### Format Ablation (Main Ablation)

Compare how task modality direction affects multi-event causal understanding:

| Exp | Training Tasks | problem_type | Research Question |
|---|---|---|---|
| **Exp A** | V→T only | `add`, `replace`, `sort` | Baseline: visual sequence → language prediction |
| **Exp B** | T→V only | `add_t2v`, `replace_t2v` | Text sequence → video recognition |
| **Exp C** | V→T + T→V mixed | all 5 types | Bidirectional alignment: synergy effect? |

### Expected findings

- **Exp A vs Exp B**: isolates visual causal reasoning from semantic causal reasoning
- **Exp C > max(A, B)**: expected if dual-direction alignment helps cross-modal grounding
- **Eval metric**: multi-event understanding benchmarks (NExT-QA, ActivityNet-QA, EgoTaskQA)

### Data balance for Exp C (Mixed)

```bash
# Combine V→T and T→V data at 1:1 ratio
python -c "
import json, random

vt = list(open('data/proxy_train_text_options.jsonl'))
tv = list(open('data/l2_event_logic_t2v_filtered.jsonl'))

# Balance to 500 each = 1000 total
rng = random.Random(42)
combined = rng.sample(vt, min(500, len(vt))) + rng.sample(tv, min(500, len(tv)))
rng.shuffle(combined)

with open('data/mixed_vt_tv_1000.jsonl', 'w') as f:
    for l in combined: f.write(l)
print(f'Mixed dataset: {len(combined)} samples')
"
```

### Task weight configuration for Exp C

Since `add_t2v` and `replace_t2v` use the same MCQ reward as `add` and `replace`, no reward function changes are needed. Just set task weights in training config:

```yaml
# Example config snippet for Exp C
task_weights:
  add: 0.2
  replace: 0.2
  sort: 0.1
  add_t2v: 0.25
  replace_t2v: 0.25
```

---

## Reward Function Update (Required)

Before training, add `add_t2v` and `replace_t2v` to `verl/reward_function/mixed_proxy_reward.py`:

```python
# In the reward dispatch dict, add:
"add_t2v":     _mcq_reward,   # same MCQ exact-match as "add"
"replace_t2v": _mcq_reward,   # same MCQ exact-match as "replace"
```

The answer format is identical to V→T: `<answer>B</answer>`.

---

## Key Design Decisions

1. **Caption style**: recipe instruction (imperative, action-focused) rather than AoT-style temporal-direction captions. This gives cleaner causal context for T→V reasoning.

2. **Negative video sampling**: same 1/3 cross-video + 2/3 same-video logic as V→T text negatives. Same-video clips are harder distractors (similar kitchen, similar ingredients).

3. **Sort task not extended to T→V**: Sort is inherently a visual reordering task; a text→video-ordering variant would not be symmetric or natural.

4. **AI causality filter for T→V**: checks (a) text context has clear causal flow AND (b) exactly one video option is unambiguously correct. More expensive than V→T filter (requires decoding option video frames), use `--filter-workers 8` to parallelize.

5. **clip_key alignment**: `annotate_l2_step_captions.py --from-dataset` uses `Path(video_path).stem` as clip_key, which matches `aot_event_manifest.jsonl`'s `clip_key` field exactly. No custom mapping needed.
