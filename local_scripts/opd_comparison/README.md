# OPD Comparison Settings

These scripts run full-composition data settings for GRPO/MOPD comparisons.

Shared defaults:

- Data: `composition_base_seg_logic_aot_hier10k_el10k_aot10k_mf256_ema`
- Full epoch: `TOTAL_EPOCHS=1`, no `MAX_STEPS` unless `ALLOW_MAX_STEPS_OVERRIDE=true`
- 8 GPUs: `N_GPUS_PER_NODE=8`
- 4B EMA-GRPO rerun batch: `ROLLOUT_BS=64`, `GLOBAL_BS=64`, `VAL_BATCH_SIZE=64`
- MOPD batch: `ROLLOUT_BS=64`, `GLOBAL_BS=64`, `VAL_BATCH_SIZE=64` for both 4B and 8B students
- 4B comparison runs save every 50 steps with no checkpoint pruning: `SAVE_FREQ=50`, `SAVE_LIMIT=-1`
- 8B MOPD saves every 50 steps but keeps only the latest regular checkpoint plus the best validation checkpoint: `SAVE_LIMIT=1`, `SAVE_BEST=true`
- 8B MOPD disables the training GPU filler by default: `ENABLE_GPU_FILLER=false`
- Checkpoints:
  - 4B: `/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/opd_comparison_4b`
  - 8B: `/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/opd_comparison_8b`
  - 8B MOPD default run: `/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/opd_comparison_8b/mopd_qwen3vl8b_full_comp_4b_teachers_bs64_mf256_epoch1_save50_keep1`
- MOPD teachers:
  - AoT: `composition_base_aot_aot10k_mf256_ema/global_step_200`
  - Seg: `composition_base_seg_hier10k_mf256_ema/global_step_250`
  - Event logic: `composition_base_logic_el10k_mf256_ema/global_step_272`

Run the 8B student MOPD setting:

```bash
bash local_scripts/opd_comparison/run_mopd_8b_from_4b_teachers.sh
```

Run the 4B full-data EMA-GRPO rerun:

```bash
bash local_scripts/opd_comparison/run_grpo_4b_full_epoch.sh
```

Run the 4B full-data MOPD setting:

```bash
bash local_scripts/opd_comparison/run_mopd_4b_full_epoch.sh
```
