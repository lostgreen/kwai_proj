# Qwen3-VL-4B 100-step CoT Budget Report

Date: 2026-05-18

## Scope

This report analyzes the three completed 100-step CoT teacher runs:

- `qwen3_vl_4b_aot_100step_cot`
- `qwen3_vl_4b_seg_100step_cot`
- `qwen3_vl_4b_logic_100step_cot`

Common setup:

- `MAX_STEPS=100`
- `VAL_FREQ=25`
- `SAVE_FREQ=100`
- `VAL_BEFORE_TRAIN=true`
- `MAX_RESPONSE_LEN=256`
- `COT_BUDGET_MAX_TOKENS=128`
- CoT tag: `<thought>...</thought>`

Remote checkpoint root:

```text
/m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task
```

Each experiment contains `experiment_log.jsonl` with 101 rows, rollout files for steps 1-100, and validation rollout files at steps 0/25/50/75/100.

## Executive Summary

The 128-token CoT budget is workable for these 4B teachers. The models learn to emit short tagged reasoning and do not collapse into repeated templates. However, the three runs differ sharply:

- Logic is the cleanest: highest final val reward, lowest KL, shortest CoT, and near-perfect CoT format.
- Seg is also stable: strong task learning and almost no budget repair at val100.
- AOT learns useful task behavior, but its task-specific CoT often runs into the budget repair path. This is the only run where CoT quality is still a real concern.

The case review suggests the models are not merely gaming the format reward. Many high-reward CoTs cite concrete temporal evidence and use compact task-specific reasoning. The weaker cases are mostly wrong visual/temporal judgments despite valid formatting, rather than empty format-only strings. AOT has the most budget-pressure cases, where reasoning is long and sometimes truncated before it can finish cleanly.

## Main Metrics

Validation reward and CoT quality from `val_step_000100.jsonl`:

| Run | Val reward 0 -> 100 | Base reward 0 -> 100 | CoT format 0 -> 100 | Val100 start | Val100 end | Val100 repair | Val100 final len mean | Val100 words mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AOT | 0.3231 -> 0.5292 | 0.5056 -> 0.5795 | 0.7563 -> 0.9137 | 1.000 | 0.982 | 0.173 | 96.34 | 68.2 |
| Seg | 0.2708 -> 0.5452 | 0.4356 -> 0.5472 | 0.7430 -> 0.9963 | 1.000 | 0.997 | 0.007 | 76.11 | 50.8 |
| Logic | 0.3269 -> 0.5765 | 0.4902 -> 0.5786 | 0.7657 -> 0.9953 | 1.000 | 0.998 | 0.009 | 69.75 | 48.4 |

Training scalar highlights at step 100:

| Run | Train reward overall | Val reward | KL loss | Response clip ratio | Response mean len |
| --- | ---: | ---: | ---: | ---: | ---: |
| AOT | 0.5354 | 0.5292 | 0.2088 | 0.000 | 114.63 |
| Seg | 0.5254 | 0.5452 | 0.1519 | 0.000 | 108.08 |
| Logic | 0.5983 | 0.5765 | 0.0989 | 0.000 | 88.41 |

Interpretation:

- None of the runs are hitting `MAX_RESPONSE_LEN=256` at step 100.
- Logic is the most efficient: it gets the best val reward with the shortest outputs and lowest KL.
- AOT has higher KL and much higher repair ratio, so it is less stable under the same CoT budget.

## Task-Level CoT Quality at Val100

The table below uses `cot_ok = start tag detected + end tag detected + not repaired`.

| Run | Task | n | Reward | CoT ok | Repair | Avg thought words |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AOT | llava_mcq | 600 | 0.7792 | 0.9183 | 0.0817 | 57.9 |
| AOT | temporal_grounding | 600 | 0.2930 | 0.9283 | 0.0717 | 49.0 |
| AOT | seg_aot_action_t2v_binary | 148 | 0.4966 | 0.4797 | 0.5203 | 96.6 |
| AOT | seg_aot_action_v2t_3way | 69 | 0.4130 | 0.3768 | 0.6232 | 94.2 |
| AOT | seg_aot_event_t2v_binary | 72 | 0.6181 | 0.4444 | 0.5556 | 98.8 |
| AOT | seg_aot_event_v2t_3way | 11 | 0.3636 | 0.3636 | 0.6364 | 91.0 |
| Seg | llava_mcq | 600 | 0.8175 | 0.9883 | 0.0117 | 40.6 |
| Seg | temporal_grounding | 600 | 0.3020 | 0.9950 | 0.0050 | 40.2 |
| Seg | temporal_seg_hier_L1 | 30 | 0.4831 | 1.0000 | 0.0000 | 58.8 |
| Seg | temporal_seg_hier_L2 | 60 | 0.5079 | 1.0000 | 0.0000 | 63.0 |
| Seg | temporal_seg_hier_L3_seg | 60 | 0.3233 | 1.0000 | 0.0000 | 66.1 |
| Logic | llava_mcq | 600 | 0.8150 | 0.9917 | 0.0083 | 38.0 |
| Logic | temporal_grounding | 600 | 0.3011 | 0.9983 | 0.0017 | 36.0 |
| Logic | event_logic_fill_blank | 138 | 0.6630 | 0.9855 | 0.0145 | 61.5 |
| Logic | event_logic_predict_next | 124 | 0.6371 | 0.9758 | 0.0242 | 61.1 |
| Logic | event_logic_sort | 38 | 0.6458 | 0.9211 | 0.0789 | 74.1 |

Findings:

- Seg and Logic CoTs are usually compact and stable. Their task-specific repair rates are near zero except logic sort, which still remains acceptable.
- AOT task-specific CoTs are substantially longer and often repaired. This is the strongest evidence that AOT is straining the 128-token budget.
- AOT base tasks, TG and MCQ, are mostly fine. The instability is concentrated in the AOT proxy tasks.

## Learning Curves

Validation reward by checkpoint:

| Run | Step 0 | Step 25 | Step 50 | Step 75 | Step 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AOT | 0.3231 | 0.5257 | 0.5437 | 0.5279 | 0.5292 |
| Seg | 0.2708 | 0.4805 | 0.5250 | 0.5392 | 0.5452 |
| Logic | 0.3269 | 0.5346 | 0.5634 | 0.5756 | 0.5765 |

CoT format reward by checkpoint:

| Run | Step 0 | Step 25 | Step 50 | Step 75 | Step 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AOT | 0.7563 | 0.9523 | 0.9820 | 0.9030 | 0.9137 |
| Seg | 0.7430 | 0.9578 | 0.9944 | 0.9963 | 0.9963 |
| Logic | 0.7657 | 0.9760 | 0.9950 | 0.9957 | 0.9953 |

The AOT curve is telling: task accuracy continues to improve, but CoT format regresses after step 50. That is consistent with the high repair rate in AOT task-specific val cases.

## Case Analysis: Is CoT Real or Format Farming?

The val files do not share temporal grounding `uid`s across the three runs, so direct same-sample cross-model comparison was not possible. The case analysis below therefore uses representative samples from each val100 file.

### AOT: Good CoT

Case: `seg_aot_action_v2t_3way`, reward 1.0, `cot_ok=true`, not repaired.

Prompt asks which paragraph matches the observed temporal order. The model writes:

```text
1. The video starts with a close-up of the athlete raising his arms in celebration (0:01-0:02).
2. Then, the athlete is shown holding his medal and smiling (0:03-0:04).
3. Next, an official places a medal around the athlete's neck (0:05-0:06).
4. Finally, the athlete hugs the official on the podium (0:07-0:09). This sequence matches option C.
```

Answer: `<answer>C</answer>`, GT: `C`.

Assessment: this is genuine temporal reasoning. It cites ordered visual events and approximate timestamps, then maps them to an option. It is concise enough and not just a format shell.

### AOT: Format-Valid But Wrong

Case: `seg_aot_action_t2v_binary`, reward 0.0, `cot_ok=true`, not repaired.

The model writes:

```text
1. The paragraph describes a sequence of actions: spinning a mop, looking at clean hands, lifting the mop, and then displaying a logo.
2. Clip A shows a woman in a red dress using a foot pedal to spin a mop, then looking at her hands, lifting the mop, and finally showing the logo.
3. Clip B does not show these specific actions in the described order.
4. Therefore, Clip A matches the paragraph.
```

Answer: `<answer>A</answer>`, GT: `B`.

Assessment: this is not empty format farming; it is a plausible but wrong visual comparison. The CoT has structure and task grounding, but the visual discrimination is wrong. This means format reward alone is not sufficient, but the failure mode is still semantic, not purely syntactic.

### AOT: Budget Pressure

Case: `seg_aot_action_t2v_binary`, reward 0.5, `cot_ok=false`, repaired.

The thought starts comparing the clips but gets cut:

```text
1. The paragraph describes a sequence of actions: inspecting a double-stitched seam, sewing another seam, inspecting the newly sewn seam, and then sewing a curved seam.
2. Clip A shows a continuous process of sewing, with the hands guiding the fabric under the needle and adjusting the fabric as needed. It starts with a seam being sewn, then shows the inspection of the seam, followed by another seam being sewn, and then a curved seam.
3. Clip B, on the other hand, appears to show a different sequence, with the ha...
```

Answer: `<answer>A</answer>`, GT: `A`.

Assessment: this is the pattern behind AOT's high repair rate. The model is doing comparison, but writes too much before deciding. It can still answer correctly, but the reasoning is not budget-disciplined.

### Seg: Good CoT

Case: `temporal_seg_hier_L2`, reward 1.0, `cot_ok=true`, not repaired.

```text
The video starts with the man speaking outdoors, then transitions to a close-up of car parts being worked on. The shots are distinct in setting and activity, so each segment is kept separate. The man is speaking and gesturing, then the camera shifts to show mechanical work on the car's undercarriage. These are clear changes in subject and focus, justifying separate events.
```

Answer: `<events>[[0, 88], [88, 171]]</events>`, GT: `<events>[[0, 88], [88, 171]]</events>`.

Assessment: this is a good compact segmentation rationale. It names the transition criterion, not just the output format.

### Seg: Format-Valid But Under-Segmented

Case: `temporal_seg_hier_L3_seg`, reward 0.0498, `cot_ok=true`, not repaired.

```text
The video starts with a close-up of hands planting a seedling into a pot, which continues for several seconds. Then, the camera pulls back to show the person standing in a greenhouse with multiple plants around her. These are distinct visual changes: one is a close-up action, and the other is a wider shot showing the environment. Therefore, I will split the video at the point where the camera angle changes.
```

Answer: `<events>[[0, 58], [58, 63]]</events>`

GT: `<events>[[5, 8], [9, 13], [14, 17], [18, 22], [23, 36], [37, 46], [47, 57]]</events>`

Assessment: this is informative but too coarse. The model learned a reasonable high-level segmentation heuristic but misses fine-grained L3 boundaries. This is not format farming; it is a granularity error.

## L3 Annotation Audit: Why Seg L3 May Look Too Coarse

After reviewing the original hierarchical segmentation annotations under:

```text
/m2v_intern/xuboshen/zgw/data/VideoProxyMixed/hier_seg_annotation_v1
```

the strongest finding is that the original L3 annotation is indeed more specific than the current training prompt/view makes obvious.

Original L3 annotation was scene-aware:

- The annotation prompt explicitly says frames labeled `[SCENE BREAK]` mark detected shot/scene boundaries.
- For a multi-scene merged L2 event, it asks for at least one L3 `sub_action` per original scene, and finer splits if the scene itself is splittable.
- The L3 log files confirm that the VLM saw frame labels such as `[SCENE BREAK] [t=17s]`, `[SCENE BREAK] [t=23s]`, etc.

Training L3 records do not preserve that supervision channel:

- `build_l3_seg_records` converts `level3.grounding_results[].sub_actions` into relative `<events>[[start, end], ...]</events>` spans.
- Metadata keeps `clip_key`, `parent_event_id`, event/clip range, and `n_actions`, but not `scene_ids` or explicit scene-break timestamps.
- The training prompt says "SHOT-FIRST", but the input is just sampled frames plus generic text. It does not show `[SCENE BREAK]` labels or the rule "multi-scene event -> at least one segment per scene".
- The current L3 prompt also contains a tension: it says camera cuts are primary anchors, while the sparse-sampling notice says not to rely on camera cuts to place boundaries. This can push the model toward coarser, more sustained state/task changes.

This suggests a prompt/data-view mismatch: the original labels encode "shot-first, then split long shots by state/action change", but the trained model is asked to infer that from sparse frames without the same explicit shot-boundary hints.

### Original L3 Case 1: Multi-Shot Event Split One Per Scene

Annotation file: `--c9_lXDKXU.json`, event 2, range 6-55s.

L2 event:

- `scene_ids`: `[2, 3, 4, 5, 6]`
- instruction: "A woman introduces the ricotta cake recipe while sitting at her desk."
- dense caption: the shot briefly cuts to the finished cake slice, then returns to the woman speaking.

L3 sub-actions:

| id | time | sub-action |
| ---: | --- | --- |
| 1 | 6-16 | A woman sits at a desk and speaks to the camera. |
| 2 | 17-22 | A slice of cake sits on a plate, with the rest of the cake in the background. |
| 3 | 23-36 | The woman continues speaking to the camera at her desk. |
| 4 | 37-39 | A close-up shot of the woman speaking. |
| 5 | 40-54 | The woman speaks and gestures from her desk. |

The L3 log for this event shows `n_scenes=5` and frame labels with scene breaks at 6, 17, 23, 37, 40, and 55 seconds. This is almost exactly "one segment per scene/shot".

### Original L3 Case 2: Multi-Shot Event Split Finer Than Scenes

Annotation file: `--c9_lXDKXU.json`, event 4, range 71-98s.

L2 event:

- `scene_ids`: `[8, 9, 10]`
- instruction: "Combine egg yolks, ricotta, stevia, flour, vanilla, and lemon zest in a bowl and mix."

L3 sub-actions:

| id | time | sub-action |
| ---: | --- | --- |
| 1 | 71-78 | A person adds ricotta cheese to a bowl of egg yolks. |
| 2 | 79-81 | A person pours white granulated sweetener into the bowl. |
| 3 | 82-84 | A person adds flour to the bowl. |
| 4 | 85-86 | A person adds vanilla essence to the bowl. |
| 5 | 87-91 | A person grates lemon zest into the bowl. |
| 6 | 92-97 | A person mixes all the ingredients together with a spatula. |

Here L3 is finer than shot count. It treats visible ingredient/state changes inside the broader event as separate micro-actions.

### Original L3 Case 3: Single-Shot Long Action Still Split

Annotation file: `-1H5IyO1MNY.json`, event 3, range 31-57s.

L2 event:

- `scene_ids`: `[2]`
- instruction: "The bartender pours red liqueur and then carefully layers green liqueur on top using a spoon."

L3 sub-actions:

| id | time | sub-action |
| ---: | --- | --- |
| 1 | 31-33 | Bartender pours red liquid into a shot glass. |
| 2 | 34-40 | Bartender picks up a green bottle and a bar spoon. |
| 3 | 41-56 | Bartender pours green liquid over a spoon into the glass. |

This confirms that original L3 is not only shot segmentation. For long or continuous shots, it further splits by material/object state and action phase.

### Original L3 Case 4: Legitimate Exception, Not Pure Shot Count

Annotation file: `-QDiVQiCa50.json`, event 1, range 0-9s.

L2 event:

- `scene_ids`: `[1, 2]`
- instruction: "A title card for a Bosch power tool is shown with the text '36V: Light Speed Power. Can you handle it?'."

L3 sub-actions:

| id | time | sub-action |
| ---: | --- | --- |
| 1 | 1-8 | A promotional graphic for a Bosch power tool is displayed. |

This is a reasonable exception: two detected scenes collapse into one static/promotional visual unit. The annotation prompt allows empty or compact output for static content, so the rule is not a blind "number of scenes equals number of segments" constraint.

### Implication For The Current Seg Model

The under-segmented val case is consistent with this mismatch. The model's CoT says it found a close-up planting shot and a wider greenhouse shot, then outputs two broad segments. That behavior is understandable under the generic training prompt, but it is too coarse relative to original L3 labels, where the target often preserves shot boundaries and also splits within long shots by visible state/action changes.

Recommended next test: run a small prompt/data ablation for L3 only:

- add scene-break timestamps or frame labels to L3 training/inference records when available;
- add the explicit rule "multi-shot event: preserve each visually distinct shot as at least one segment unless it is static/redundant";
- remove or soften the sparse-sampling sentence that says not to rely on camera cuts;
- track under-segmentation separately with predicted segment count vs GT segment count, not only reward.

### Logic: Good CoT

Case: `event_logic_predict_next`, reward 1.0, `cot_ok=true`, not repaired.

```text
The person has just drained the lentils, which means the next logical step is to refill the bowl with water, as this is typically done before cooking lentils to soak them.
```

Answer: `<answer>E</answer>`, GT: `E`.

Assessment: this is exactly what we want under a small budget: one visual observation, one causal/process inference, one answer.

### Logic: Format-Valid But Wrong

Case: `event_logic_fill_blank`, reward 0.0, `cot_ok=true`, not repaired.

```text
The sequence shows a machine cutting into a rotating metal piece, with metal shavings being produced. This is indicative of continuous cutting action. The option that best fits this continuous process is B.
```

Answer: `<answer>B</answer>`, GT: `C`.

Assessment: concise and grounded, but chooses the wrong option. Again, this looks like a semantic miss rather than a hollow format trick.

### Logic: Budget Repair Case

Case: `event_logic_predict_next`, reward 0.5, repaired.

```text
The video shows the process of cooking minced garlic in oil, then adding a dark liquid (likely wine or broth) and stirring, followed by the addition of a red liquid (likely tomato sauce or another condiment), and stirring again. The mixture is reduced and thickened. This is a typical preparation of a sauce or reduction, which is often used to coat or accompany cooked proteins. The logical next step in such a cooking process would be to add the cooked meat, which in this context, given the common pairing of garli...
```

Answer: empty, GT: `F`.

Assessment: this is a clear budget failure. The reasoning is rich, but it spends the budget before producing the final answer. This is rare in Logic at val100, but it shows why repair ratio should stay monitored.

## Overall Judgment on CoT Quality

The CoT is mostly real, not just format padding.

Evidence:

- High-reward cases contain task-specific observations: temporal order, scene transitions, option comparison, process causality.
- Low-reward but format-valid cases are often plausible wrong interpretations, not generic "I analyze the video" filler.
- Logic and Seg are able to compress useful reasoning into 35-70 words.
- The main negative signal is not hollow CoT, but AOT budget pressure and Seg L3 under-segmentation.

That said, format reward can mask partial failures:

- A format-valid CoT can still be semantically wrong.
- A repaired CoT can still receive partial task reward if the answer survives.
- AOT proxy tasks show many repaired CoTs, so their final score mixes real task learning with unstable reasoning style.

## Recommendations

1. Continue Logic and Seg longer.
   Their CoT behavior is stable under 128 tokens. Logic especially looks ready for a longer run.

2. For AOT, run a short budget ablation before scaling.
   Suggested variants:
   - keep 128 tokens but add a shorter AOT-specific CoT instruction;
   - compare `COT_BUDGET_MAX_TOKENS=160` or `192`;
   - monitor task-specific `cot_ok` rather than only global `cot_format_reward`.

3. Add a CoT quality audit script.
   It should report per task:
   - reward bucket x `cot_ok`;
   - repaired examples;
   - thought word/token length distribution;
   - low-reward-but-format-ok cases.

4. For Seg L3, evaluate granularity separately.
   The model's CoT can be meaningful while the output is too coarse. A dedicated L3 boundary-count or over/under-segmentation metric would make this clearer than reward alone.

5. For Seg L3, align training prompt with original scene-first annotation.
   The original L3 labels were produced with `[SCENE BREAK]` cues and an explicit multi-scene minimum rule. The current L3 training prompt keeps the words "SHOT-FIRST" but removes those cues. This likely encourages broad two-part segmentations when the model should preserve shot-level anchors and then split long shots by state/action changes.

## Appendix: Commands Used

Read-only remote analysis used:

```text
python3 /home/xuboshen/zgw/EasyR1/video_proxy/training/debug/analyze_train_step.py \
  /m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_aot_100step_cot \
  /m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_seg_100step_cot \
  /m2v_intern/xuboshen/zgw/RL-Models/VideoProxyMixed/multi_task/qwen3_vl_4b_logic_100step_cot \
  --start 1 --step 100 --rollout-steps 1,25,50,75,100
```

Additional read-only Python snippets parsed:

- `experiment_log.jsonl`
- `rollouts/val_step_000100.jsonl`
- `cot_budget_debug`
- `<thought>...</thought>` spans
