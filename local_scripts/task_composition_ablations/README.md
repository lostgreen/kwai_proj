# Task Composition Ablations

This folder runs the current fast 4B EMA-GRPO task-composition sweep:

| Script | Tasks | Default EXP_NAME |
| --- | --- | --- |
| `exp_base_seg.sh` | `tg mcq hier_seg` | `composition_base_seg_hier10k_mf256_ema` |
| `exp_base_aot.sh` | `tg mcq aot` | `composition_base_aot_aot10k_mf256_ema` |
| `exp_base_seg_aot.sh` | `tg mcq hier_seg aot` | `composition_base_seg_aot_hier10k_aot10k_mf256_ema` |

Shared defaults:

- Model: `Qwen3-VL-4B-Instruct`
- Algorithm: `ema_grpo`
- `KL_COEF=0.01`, `LR=5e-7`, `ENTROPY_COEFF=0.005`
- `MAX_FRAMES=256`, `MAX_PIXELS=65536`
- `HIER_TARGET=10000`, `AOT_TARGET=10000`
- checkpoint root: `/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task_4b_lr5e-7_kl0p01_entropy0p005_ablations`

Run one experiment:

```bash
bash local_scripts/task_composition_ablations/exp_base_seg.sh
bash local_scripts/task_composition_ablations/exp_base_aot.sh
bash local_scripts/task_composition_ablations/exp_base_seg_aot.sh
```

Run all three sequentially:

```bash
bash local_scripts/task_composition_ablations/run_composition_ablations.sh
```

Limit to a subset:

```bash
EXPS="BASE_SEG BASE_AOT" bash local_scripts/task_composition_ablations/run_composition_ablations.sh
```
