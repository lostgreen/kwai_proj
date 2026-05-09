# Training Tools

Utilities for checkpoint management, rollout inspection, GPU filler behavior,
and training diagnostics.

## Examples

```bash
bash video_proxy/training/tools/cleanup_checkpoints.sh /path/to/checkpoints
python video_proxy/training/tools/probe_cot_tags.py --help
python video_proxy/training/tools/sample_rollout_analysis.py --help
```

Long-running training entrypoints belong in `video_proxy/experiments/`, not here.
