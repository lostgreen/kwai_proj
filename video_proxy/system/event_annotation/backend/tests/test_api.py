import json
import os
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

def _setup_app(tmp_dir: str):
    os.environ["VIZ_EXAMPLES_DIR"] = tmp_dir
    ex_dir = Path(tmp_dir) / "test_ex"
    ex_dir.mkdir(parents=True)
    (ex_dir / "analysis.json").write_text(json.dumps({
        "video_id": "test_ex", "video_url": "/videos/test_ex/video.mp4",
        "duration": 30.0, "caption": "Test",
        "hierarchy": {"L1_phases": [], "L2_events": [], "L3_actions": []},
        "causal_chain": [], "key_frames": {},
    }))
    from importlib import reload
    import app.config; reload(app.config)
    import app.routers.analysis; reload(app.routers.analysis)
    import app.main; reload(app.main)
    return TestClient(app.main.app)

def test_health():
    with tempfile.TemporaryDirectory() as tmp:
        client = _setup_app(tmp)
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

def test_list_examples():
    with tempfile.TemporaryDirectory() as tmp:
        client = _setup_app(tmp)
        r = client.get("/api/examples")
        assert r.status_code == 200
        assert len(r.json()) == 1

def test_get_example():
    with tempfile.TemporaryDirectory() as tmp:
        client = _setup_app(tmp)
        r = client.get("/api/examples/test_ex")
        assert r.status_code == 200
        assert r.json()["video_id"] == "test_ex"

def test_get_example_404():
    with tempfile.TemporaryDirectory() as tmp:
        client = _setup_app(tmp)
        r = client.get("/api/examples/nonexistent")
        assert r.status_code == 404
