from fastapi import APIRouter, HTTPException
from app.config import Settings
from app.models.schemas import AnalysisResult, ExampleSummary
from app.services.example_service import ExampleService

router = APIRouter()
_settings = Settings()
_example_svc = ExampleService(_settings.examples_dir)

@router.get("/health")
def health():
    return {"status": "ok", "model": _settings.model_name}

@router.get("/examples", response_model=list[ExampleSummary])
def list_examples():
    return _example_svc.list_examples()

@router.get("/examples/{example_id}", response_model=AnalysisResult)
def get_example(example_id: str):
    result = _example_svc.get_example(example_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Example not found")
    return result
