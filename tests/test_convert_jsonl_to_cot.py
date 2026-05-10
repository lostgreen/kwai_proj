from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "video_proxy" / "data" / "scripts" / "convert_jsonl_to_cot.py"
_SPEC = importlib.util.spec_from_file_location("convert_jsonl_to_cot", _MODULE_PATH)
convert_jsonl_to_cot = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = convert_jsonl_to_cot
_SPEC.loader.exec_module(convert_jsonl_to_cot)


def test_rewrite_tg_natural_prompt_to_cot_and_syncs_last_user_message():
    prompt = (
        "<video>\n"
        "Please find the visual event described by a sentence in the video, "
        "determining its starting and ending times. "
        "Now I will give you the textual sentence: \"slice the onion\". "
        "Please return its start time and end time in seconds."
    )
    record = {
        "prompt": prompt,
        "messages": [
            {"role": "system", "content": "system stays"},
            {"role": "user", "content": prompt},
        ],
        "answer": "The event happens in the 1.2 - 3.4 seconds.",
        "problem_type": "temporal_grounding",
    }

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record)

    assert changed is True
    assert reason == "tg_natural"
    assert "First, think step by step inside <think></think>" in converted["prompt"]
    assert converted["prompt"].endswith("then give the final sentence only.")
    assert converted["answer"] == record["answer"]
    assert converted["messages"][0]["content"] == "system stays"
    assert converted["messages"][1] == {"role": "user", "content": converted["prompt"]}


def test_rewrite_choice_direct_instruction_to_cot_without_changing_answer():
    prompt = (
        "<video>\n"
        "Which option is correct?\n"
        "Options:\nA. first\nB. second\n\n"
        "Provide your answer (a single letter from A, B) inside <answer> </answer> tags."
    )
    record = {"prompt": prompt, "answer": "B", "problem_type": "event_logic_predict_next"}

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record)

    assert changed is True
    assert reason == "answer_tag"
    assert "Think step by step inside <think></think> tags" in converted["prompt"]
    assert "final answer (a single letter from A, B) inside <answer></answer> tags" in converted["prompt"]
    assert converted["answer"] == "B"
    assert converted["messages"] == [{"role": "user", "content": converted["prompt"]}]


def test_rewrite_choice_direct_instruction_can_use_thought_tag():
    prompt = (
        "<video>\n"
        "Which option is correct?\n"
        "Options:\nA. first\nB. second\n\n"
        "Provide your answer (a single letter from A, B) inside <answer> </answer> tags."
    )
    record = {"prompt": prompt, "answer": "B", "problem_type": "event_logic_predict_next"}

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record, reasoning_tag="thought")

    assert changed is True
    assert reason == "answer_tag"
    assert "Think step by step inside <thought></thought> tags" in converted["prompt"]
    assert "<think>" not in converted["prompt"]


def test_rewrite_aot_only_one_letter_instruction_to_cot_answer_tags():
    prompt = (
        "<video><video>\n\n"
        "Here are two video clips (Clip A and Clip B).\n\n"
        "Which clip (A or B) shows these events in the listed order?\n\n"
        "Answer with only one letter: A or B."
    )
    record = {"prompt": prompt, "answer": "A", "problem_type": "seg_aot_action_t2v_binary"}

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record, reasoning_tag="thought")

    assert changed is True
    assert reason == "answer_tag"
    assert "Think step by step inside <thought></thought> tags" in converted["prompt"]
    assert "then provide your final answer (a single letter: A or B) inside <answer></answer> tags." in converted["prompt"]
    assert "Answer with only one letter" not in converted["prompt"]
    assert converted["messages"] == [{"role": "user", "content": converted["prompt"]}]


def test_rewrite_aot_three_way_only_one_letter_instruction_to_cot_answer_tags():
    prompt = (
        "<video>\n"
        "Which option lists the events in chronological order?\n\n"
        "A. first option\n"
        "B. second option\n"
        "C. third option\n\n"
        "Answer with only one letter: A, B, or C."
    )
    record = {"prompt": prompt, "answer": "C", "problem_type": "seg_aot_event_v2t_3way"}

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record, reasoning_tag="thought")

    assert changed is True
    assert reason == "answer_tag"
    assert "final answer (a single letter: A, B, or C) inside <answer></answer> tags." in converted["prompt"]
    assert "Answer with only one letter" not in converted["prompt"]


def test_rewrite_sort_direct_instruction_keeps_sequence_answer_detail():
    prompt = (
        "<video>\n"
        "Determine the correct chronological order of these clips.\n\n"
        "Provide your answer as a sequence of clip numbers with no spaces or separators "
        "(e.g., 321) inside <answer> </answer> tags."
    )
    record = {"prompt": prompt, "answer": "321", "problem_type": "event_logic_sort"}

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record)

    assert changed is True
    assert reason == "answer_tag"
    assert "Think step by step inside <think></think> tags" in converted["prompt"]
    assert "final answer as a sequence of clip numbers with no spaces or separators" in converted["prompt"]
    assert "(e.g., 321) inside <answer></answer> tags" in converted["prompt"]
    assert converted["answer"] == "321"


def test_rewrite_events_prompt_inserts_cot_before_final_output_instruction():
    prompt = (
        "<video>\n"
        "Segment the video into phases.\n\n"
        "Output the start and end time (integer seconds, 0-based) for each phase in chronological order:\n"
        "<events>[[start_time, end_time], ...]</events>\n\n"
        "Example: <events>[[0, 10], [10, 20]]</events>"
    )
    record = {"prompt": prompt, "answer": "<events>[[0, 10], [10, 20]]</events>"}

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record)

    assert changed is True
    assert reason == "events"
    assert converted["prompt"].count("<think></think>") == 1
    assert converted["prompt"].index("First, think step by step") < converted["prompt"].index("Output the start")
    assert converted["answer"] == record["answer"]


def test_rewrite_shot_first_events_prompt_uses_cot_example_not_final_only_example():
    prompt = (
        "<video>\n"
        "Detect all events in this clip using a SHOT-FIRST approach:\n\n"
        "STEP 1 — IDENTIFY SHOTS:\n"
        "Scan the clip for visual shot transitions.\n\n"
        "STEP 2 — RESTRUCTURE INTO EVENTS using three operations:\n"
        "- KEEP: A single shot that contains one coherent task becomes one event.\n"
        "- MERGE: Consecutive shots that belong to the same local task become one event.\n"
        "- SPLIT: A long single-shot segment that contains multiple distinct tasks should be split.\n\n"
        "PARTITION RULE: The events must form a non-overlapping, gap-free partition of the full clip timeline.\n\n"
        "Output the start and end time (integer seconds, 0-based) for each event in chronological order:\n"
        "<events>[[start_time, end_time], ...]</events>\n\n"
        "Example: <events>[[0, 42], [42, 68], [68, 90]]</events>"
    )
    record = {"prompt": prompt, "answer": "<events>[[0, 42], [42, 68], [68, 90]]</events>"}

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record, reasoning_tag="thought")

    assert changed is True
    assert reason == "events"
    assert "First, think step by step inside <thought></thought> tags" in converted["prompt"]
    example = converted["prompt"].split("Example:", 1)[1]
    assert "The video has five shots" in example
    assert "[0,8] oil is poured into a pan" in example
    assert "[22,34] green leaves are added followed by chopped chilies" in example
    assert "Shots [0,8], [8,14], and [14,22] are merged into [0,22]" in example
    assert "The shot [22,34] is split into [22,28] and [28,34]" in example
    assert "The shot [34,42] is kept as [34,42]" in example
    assert "cover the full 0-42s clip" in example
    assert "Example: <events>" not in converted["prompt"]
    assert "<thought>\nThe video has five shots:" in converted["prompt"]
    assert "placeholder" not in converted["prompt"].lower()
    assert "..." not in example
    assert "\nI " not in example
    assert "</thought>\n<events>[[0, 22], [22, 28], [28, 34], [34, 42]]</events>" in converted["prompt"]


def test_rewrite_generic_events_prompt_inserts_cot_before_output_format_and_scopes_timestamp_rule():
    prompt = (
        "<video>\n"
        "Segment the clip into steps.\n\n"
        "Output format (strictly follow this):\n"
        "<events>\n"
        "[start1, end1]\n"
        "</events>\n\n"
        "Rules:\n"
        "- Output only timestamps, no descriptions.\n"
        "- Timestamps must be in chronological order."
    )
    record = {"prompt": prompt, "answer": "<events>\n[0, 10]\n</events>"}

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record)

    assert changed is True
    assert reason == "events"
    assert converted["prompt"].index("First, think step by step") < converted["prompt"].index("Output format")
    assert "In the final <events> block, output only timestamps" in converted["prompt"]
    assert "- Output only timestamps, no descriptions." not in converted["prompt"]


def test_existing_cot_prompt_is_left_unchanged():
    prompt = (
        "<video>\n"
        "Think step by step inside <think></think> tags, then provide your final answer "
        "inside <answer></answer> tags."
    )
    record = {"prompt": prompt, "messages": [{"role": "user", "content": prompt}], "answer": "A"}

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record)

    assert changed is False
    assert reason == "already_cot"
    assert converted == record


def test_existing_thought_prompt_is_left_unchanged():
    prompt = (
        "<video>\n"
        "Think step by step inside <thought></thought> tags, then provide your final answer "
        "inside <answer></answer> tags."
    )
    record = {"prompt": prompt, "messages": [{"role": "user", "content": prompt}], "answer": "A"}

    converted, changed, reason = convert_jsonl_to_cot.convert_record(record)

    assert changed is False
    assert reason == "already_cot"
    assert converted == record


def test_convert_jsonl_writes_records_and_summary(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    rows = [
        {
            "prompt": "<video>\nQuestion?\n\nAnswer with the option letter.",
            "answer": "A",
            "problem_type": "llava_mcq",
        },
        {
            "prompt": "<video>\nThink step by step inside <think></think> tags. <answer></answer>",
            "answer": "B",
        },
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = convert_jsonl_to_cot.convert_jsonl(input_path, output_path)

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary.total == 2
    assert summary.converted == 1
    assert summary.unchanged == 1
    assert "Think step by step inside <think></think> tags" in written[0]["prompt"]
    assert written[1] == rows[1]


def test_convert_jsonl_uses_custom_reasoning_tag(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    rows = [
        {
            "prompt": "<video>\nQuestion?\n\nAnswer with the option letter.",
            "answer": "A",
            "problem_type": "llava_mcq",
        },
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = convert_jsonl_to_cot.convert_jsonl(input_path, output_path, reasoning_tag="thought")

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert summary.converted == 1
    assert "<thought></thought>" in written[0]["prompt"]
    assert "<think>" not in written[0]["prompt"]


def test_tg_compat_converter_delegates_to_shared_cli():
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "video_proxy"
        / "data"
        / "base_sources"
        / "tg"
        / "convert_nocot_to_cot.py"
    )

    source = wrapper.read_text(encoding="utf-8")

    assert "video_proxy.data.scripts.convert_jsonl_to_cot" in source
    assert "main()" in source


def test_collect_prompt_samples_groups_by_problem_type_and_limits_per_type(tmp_path: Path):
    path = tmp_path / "converted.jsonl"
    rows = [
        {"prompt": "tg prompt 0", "problem_type": "temporal_grounding"},
        {"prompt": "tg prompt 1", "problem_type": "temporal_grounding"},
        {"prompt": "tg prompt 2", "problem_type": "temporal_grounding"},
        {"prompt": "mcq prompt 0", "problem_type": "llava_mcq"},
        {"prompt": "missing type prompt"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    samples = convert_jsonl_to_cot.collect_prompt_samples(path, per_type=2)

    assert list(samples) == ["llava_mcq", "temporal_grounding", "unknown"]
    assert samples["temporal_grounding"] == ["tg prompt 0", "tg prompt 1"]
    assert samples["llava_mcq"] == ["mcq prompt 0"]
    assert samples["unknown"] == ["missing type prompt"]
