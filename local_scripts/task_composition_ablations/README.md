# Task Composition Ablations

This folder runs the current fast 4B EMA-GRPO task-composition sweep:

| Script | Tasks | Default EXP_NAME |
| --- | --- | --- |
| `exp_base.sh` | `tg mcq` | `composition_base_mf256_ema` |
| `exp_base_seg.sh` | `tg mcq hier_seg` | `composition_base_seg_hier10k_mf256_ema` |
| `exp_base_aot.sh` | `tg mcq aot` | `composition_base_aot_aot10k_mf256_ema` |
| `exp_base_logic.sh` | `tg mcq event_logic` | `composition_base_logic_el10k_mf256_ema` |
| `exp_base_seg_aot.sh` | `tg mcq hier_seg aot` | `composition_base_seg_aot_hier10k_aot10k_mf256_ema` |
| `exp_base_seg_logic.sh` | `tg mcq hier_seg event_logic` | `composition_base_seg_logic_hier10k_el10k_mf256_ema` |
| `exp_base_aot_logic.sh` | `tg mcq aot event_logic` | `composition_base_aot_logic_aot10k_el10k_mf256_ema` |
| `exp_base_seg_logic_aot.sh` | `tg mcq hier_seg event_logic aot` | `composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema` |

8B single-task composition variants:

| Script | Tasks | Default EXP_NAME |
| --- | --- | --- |
| `exp_base_seg_8b.sh` | `tg mcq hier_seg` | `composition_base_seg_hier10k_mf256_8b_ema` |
| `exp_base_aot_8b.sh` | `tg mcq aot` | `composition_base_aot_aot10k_mf256_8b_ema` |
| `exp_base_logic_8b.sh` | `tg mcq event_logic` | `composition_base_logic_el10k_mf256_8b_ema` |

Shared defaults:

- Model: `Qwen3-VL-4B-Instruct`
- Algorithm: `ema_grpo`
- `KL_COEF=0.01`, `LR=5e-7`, `ENTROPY_COEFF=0.005`
- `MAX_FRAMES=256`, `MAX_PIXELS=65536`
- `HIER_TARGET=10000`, `AOT_TARGET=10000`, `EL_TARGET=10000`
- checkpoint root: `/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations`

8B defaults:

- Model: `Qwen3-VL-8B-Instruct`
- `TP_SIZE=2`, `ROLLOUT_BS=32`, `GLOBAL_BS=32`, `VAL_BATCH_SIZE=32`
- GPU filler skips local GPU 5 by default: `FILLER_GPUS=0,1,2,3,4,6,7`
- Training/validation JSONL defaults read the corresponding 4B mixed-data experiment directories.
- checkpoint root: `/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_8b_lr5e-7_kl0p01_entropy0p005_ablations`

Run one experiment:

```bash
bash local_scripts/task_composition_ablations/exp_base.sh
bash local_scripts/task_composition_ablations/exp_base_seg.sh
bash local_scripts/task_composition_ablations/exp_base_aot.sh
bash local_scripts/task_composition_ablations/exp_base_logic.sh
bash local_scripts/task_composition_ablations/exp_base_seg_8b.sh
bash local_scripts/task_composition_ablations/exp_base_aot_8b.sh
bash local_scripts/task_composition_ablations/exp_base_logic_8b.sh
bash local_scripts/task_composition_ablations/exp_base_seg_aot.sh
bash local_scripts/task_composition_ablations/exp_base_seg_logic.sh
bash local_scripts/task_composition_ablations/exp_base_aot_logic.sh
bash local_scripts/task_composition_ablations/exp_base_seg_logic_aot.sh
```

Run the default seed set sequentially:

```bash
bash local_scripts/task_composition_ablations/run_composition_ablations.sh
```

Run the two missing logic-composition experiments:

```bash
EXPS="BASE_SEG_LOGIC BASE_AOT_LOGIC" bash local_scripts/task_composition_ablations/run_composition_ablations.sh
```

Run the 8B base+single-task experiments:

```bash
EXPS="BASE_SEG_8B BASE_AOT_8B BASE_LOGIC_8B" \
  bash local_scripts/task_composition_ablations/run_composition_ablations.sh
```

Run the full composition suite:

```bash
EXPS="BASE BASE_SEG BASE_AOT BASE_LOGIC BASE_SEG_AOT BASE_SEG_LOGIC BASE_AOT_LOGIC BASE_SEG_LOGIC_AOT" \
  bash local_scripts/task_composition_ablations/run_composition_ablations.sh
```
