from __future__ import annotations
from app.models.schemas import AnalysisResult
from app.services.example_service import ExampleService

class InferenceService:
    def __init__(self, example_svc: ExampleService, vllm_url: str = "") -> None:
        self._example_svc = example_svc
        self._vllm_url = vllm_url

    @property
    def is_mock(self) -> bool:
        return not self._vllm_url

    def analyze(self, video_path: str) -> AnalysisResult | None:
        examples = self._example_svc.list_examples()
        if examples:
            return self._example_svc.get_example(examples[0].id)
        return None
