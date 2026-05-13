# OPD Experiments

Run OPD training using a teacher checkpoint or task-specific teacher checkpoints.

## Models

- `qwen3_vl_4b/run.sh`
- `qwen3_vl_4b/run_mopd_3teachers.sh`
- `qwen3_vl_4b/run_mopd_2teachers.sh`
- `qwen3_vl_8b/run.sh`
- `qwen3_vl_8b/run_mopd_3teachers.sh`
- `qwen3_vl_8b/run_mopd_2teachers.sh`
- `qwen2_5_vl_3b/run.sh`
- `qwen2_5_vl_3b/run_mopd_3teachers.sh`
- `qwen2_5_vl_3b/run_mopd_2teachers.sh`
- `qwen2_5_vl_7b/run.sh`
- `qwen2_5_vl_7b/run_mopd_3teachers.sh`
- `qwen2_5_vl_7b/run_mopd_2teachers.sh`

Each script sets model defaults and sources `video_proxy/training/recipes/opd_train.sh`.

## Single-Teacher Example

```bash
TEACHER_MODEL_PATH=/path/to/teacher \
  bash video_proxy/experiments/opd/qwen3_vl_4b/run.sh
```

## Multi-Teacher Example

```bash
AOT_TEACHER_MODEL_PATH=/path/to/aot \
SEG_TEACHER_MODEL_PATH=/path/to/seg \
EVENTLOGIC_TEACHER_MODEL_PATH=/path/to/eventlogic \
OPD_TEACHER_KEY=problem_type \
bash video_proxy/experiments/opd/qwen3_vl_4b/run.sh
```

## Restored MOPD Presets

The `run_mopd_3teachers.sh` scripts use the full composition data
`composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema` with AOT, SEG,
and event-logic teachers.

The `run_mopd_2teachers.sh` scripts use the base+R1/R2 data
`composition_base_seg_aot_hier10k_aot10k_mf256_ema` with AOT and SEG teachers.

```bash
bash video_proxy/experiments/opd/qwen3_vl_4b/run_mopd_3teachers.sh
bash video_proxy/experiments/opd/qwen3_vl_4b/run_mopd_2teachers.sh
```
