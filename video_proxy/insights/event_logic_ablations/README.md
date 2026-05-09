# Event Logic Ablation Insights

Analyze saved train rollouts and count actual `problem_type` ratios for event
logic ablation runs.

## Run

```bash
bash video_proxy/insights/event_logic_ablations/run.sh
```

Use a custom experiment root:

```bash
bash video_proxy/insights/event_logic_ablations/run.sh \
  --experiment-root /path/to/multi_task \
  --output-root video_proxy/insights/event_logic_ablations/outputs/my_run
```

The script reads `rollouts/step_*.jsonl` and ignores validation rollout files.
