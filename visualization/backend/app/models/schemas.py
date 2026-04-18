from __future__ import annotations
from pydantic import BaseModel

class Phase(BaseModel):
    id: str
    label: str
    start: float
    end: float
    description: str = ""
    child_event_ids: list[str] = []

class Event(BaseModel):
    id: str
    label: str
    start: float
    end: float
    description: str = ""
    parent_phase_id: str = ""
    child_action_ids: list[str] = []

class Action(BaseModel):
    id: str
    label: str
    start: float
    end: float
    parent_event_id: str = ""

class CausalLink(BaseModel):
    from_id: str
    to_id: str
    relation: str

class KeyFrame(BaseModel):
    time: float
    url: str

class Hierarchy(BaseModel):
    L1_phases: list[Phase] = []
    L2_events: list[Event] = []
    L3_actions: list[Action] = []

class AnalysisResult(BaseModel):
    video_id: str
    video_url: str
    duration: float
    caption: str = ""
    domain: str | None = None
    hierarchy: Hierarchy
    causal_chain: list[CausalLink] = []
    key_frames: dict[str, list[KeyFrame]] = {}

class ExampleSummary(BaseModel):
    id: str
    title: str
    domain: str
    duration: float
    thumbnail_url: str = ""
