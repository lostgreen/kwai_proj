from app.models.schemas import (
    Action, CausalLink, Event, Phase, Hierarchy, AnalysisResult, ExampleSummary,
)

def test_phase_creation():
    p = Phase(id="p1", label="Prepare", start=0.0, end=42.0, description="Prep", child_event_ids=["e1"])
    assert p.id == "p1"
    assert p.end == 42.0

def test_event_creation():
    e = Event(id="e1", label="Chop", start=0.0, end=25.0, description="Chop vegs", parent_phase_id="p1", child_action_ids=["a1"])
    assert e.parent_phase_id == "p1"

def test_action_creation():
    a = Action(id="a1", label="Pick up knife", start=0.0, end=3.0, parent_event_id="e1")
    assert a.parent_event_id == "e1"

def test_causal_link():
    c = CausalLink(from_id="e1", to_id="e2", relation="Must chop before cooking")
    assert c.from_id == "e1"

def test_analysis_result_minimal():
    result = AnalysisResult(
        video_id="test_01", video_url="/videos/test/video.mp4", duration=60.0,
        caption="Test.", hierarchy=Hierarchy(), causal_chain=[], key_frames={},
    )
    assert result.duration == 60.0
    assert result.domain is None
