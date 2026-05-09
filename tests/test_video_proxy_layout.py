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
