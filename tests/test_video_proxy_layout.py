from pathlib import Path


def test_video_proxy_top_level_layout_is_intentional():
    root = Path("video_proxy")

    assert (root / "data").is_dir()
    assert (root / "experiments").is_dir()
    assert (root / "training").is_dir()
    assert (root / "insights").is_dir()

    assert not (root / "system").exists()
    assert not (root / "analysis").exists()
    assert not (root / "visualization").exists()


def test_insights_contains_reports_and_viewers():
    root = Path("video_proxy/insights")

    assert (root / "event_logic_ablations" / "analyze.py").is_file()
    assert (root / "event_logic_ablations" / "plot_actual_training_problem_type_ratio.py").is_file()
    assert (root / "event_logic_ablations" / "run.sh").is_file()
    assert (root / "hier_seg_frame_budget" / "analyze.py").is_file()
    assert (root / "rollout_viewer" / "server.py").is_file()
    assert (root / "rollout_viewer" / "index.html").is_file()
    assert (root / "data_browser" / "server.py").is_file()
    assert (root / "data_browser" / "index.html").is_file()


def test_major_video_proxy_directories_have_readmes():
    readme_dirs = [
        "video_proxy",
        "video_proxy/data",
        "video_proxy/data/mixing",
        "video_proxy/data/pipelines",
        "video_proxy/data/pipelines/proxy_construction",
        "video_proxy/data/scripts",
        "video_proxy/experiments",
        "video_proxy/experiments/teacher_train",
        "video_proxy/experiments/opd",
        "video_proxy/experiments/baselines",
        "video_proxy/experiments/baselines/grpo",
        "video_proxy/training",
        "video_proxy/training/common",
        "video_proxy/training/launchers",
        "video_proxy/training/recipes",
        "video_proxy/training/tools",
        "video_proxy/training/debug",
        "video_proxy/insights",
        "video_proxy/insights/event_logic_ablations",
        "video_proxy/insights/hier_seg_frame_budget",
        "video_proxy/insights/rollout_viewer",
        "video_proxy/insights/data_browser",
    ]

    missing = [path for path in readme_dirs if not (Path(path) / "README.md").is_file()]

    assert missing == []


def test_proxy_construction_layout_uses_paper_aligned_names():
    root = Path("video_proxy/data/pipelines/proxy_construction")

    assert (root / "annotation").is_dir()
    assert (root / "event_boundary").is_dir()
    assert (root / "event_progression").is_dir()
    assert (root / "event_relation").is_dir()


def test_proxy_construction_active_dirs_stay_focused():
    root = Path("video_proxy/data/pipelines/proxy_construction")

    expected_active_files = {
        root / "annotation" / "build_hierarchical_annotations.py",
        root / "annotation" / "annotation_schema.py",
        root / "annotation" / "detect_scene_boundaries.py",
        root / "annotation" / "extract_annotation_frames.py",
        root / "annotation" / "run_annotation_pipeline.sh",
        root / "event_boundary" / "build_boundary_frame_list_data.py",
        root / "event_boundary" / "boundary_prompts.py",
        root / "event_progression" / "build_progression_manifests.py",
        root / "event_progression" / "build_progression_frame_list_data.py",
        root / "event_progression" / "build_forward_reverse_frame_list_data.py",
        root / "event_progression" / "filter_progression_rollouts.py",
        root / "event_progression" / "run_progression_pipeline.py",
        root / "event_progression" / "merge_progression_rollout_shards.py",
        root / "event_relation" / "build_relation_frame_list_data.py",
        root / "event_relation" / "build_relation_training_mix.py",
        root / "event_relation" / "relation_prompts.py",
        root / "event_relation" / "run_event_relation_rollout.sh",
        root / "event_relation" / "run_event_relation_vlm.sh",
        root / "event_relation" / "relation_task_prompts.py",
    }
    active_dirs = [
        root / "annotation",
        root / "event_boundary",
        root / "event_progression",
        root / "event_relation",
    ]
    actual_active_files = {
        path
        for directory in active_dirs
        for path in directory.iterdir()
        if path.is_file() and path.name not in {"README.md", "__init__.py"}
    }

    assert actual_active_files == expected_active_files
def test_data_curation_pipeline_is_duration_first():
    root = Path("video_proxy/data/pipelines/data_curation")

    active_files = [
        root / "run.sh",
        root / "curation" / "README.md",
        root / "curation" / "sources.py",
        root / "curation" / "duration_filter.py",
        root / "curation" / "local_score.py",
    ]
    missing = [str(path) for path in active_files if not path.is_file()]
    assert missing == []

    removed_active_files = [
        root / "PIPELINE_REPORT.md",
        root / "et_instruct_164k" / "text_filter.py",
        root / "et_instruct_164k" / "sample_per_source.py",
        root / "timelens_100k" / "text_filter.py",
        root / "timelens_100k" / "sample_per_source.py",
    ]
    lingering = [str(path) for path in removed_active_files if path.exists()]
    assert lingering == []
