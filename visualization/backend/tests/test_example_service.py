import json
import tempfile
from pathlib import Path
from app.services.example_service import ExampleService

def _make_example_dir(tmp: Path, example_id: str = "test_01") -> Path:
    d = tmp / example_id
    d.mkdir(parents=True)
    analysis = {
        "video_id": example_id, "video_url": f"/videos/{example_id}/video.mp4",
        "duration": 60.0, "caption": "Test video", "domain": "test",
        "hierarchy": {"L1_phases": [], "L2_events": [], "L3_actions": []},
        "causal_chain": [], "key_frames": {},
    }
    (d / "analysis.json").write_text(json.dumps(analysis))
    return tmp

def test_list_examples():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_example_dir(root, "ex1")
        _make_example_dir(root, "ex2")
        svc = ExampleService(root)
        examples = svc.list_examples()
        assert len(examples) == 2

def test_get_example():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_example_dir(root, "ex1")
        svc = ExampleService(root)
        result = svc.get_example("ex1")
        assert result is not None
        assert result.video_id == "ex1"

def test_get_nonexistent():
    with tempfile.TemporaryDirectory() as tmp:
        svc = ExampleService(Path(tmp))
        assert svc.get_example("nope") is None
