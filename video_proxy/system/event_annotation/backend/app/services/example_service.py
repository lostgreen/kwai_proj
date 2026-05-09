from __future__ import annotations
import json
from pathlib import Path
from app.models.schemas import AnalysisResult, ExampleSummary

class ExampleService:
    def __init__(self, examples_dir: Path) -> None:
        self._dir = examples_dir

    def list_examples(self) -> list[ExampleSummary]:
        results = []
        if not self._dir.exists():
            return results
        for d in sorted(self._dir.iterdir()):
            analysis_file = d / "analysis.json"
            if not analysis_file.exists():
                continue
            data = json.loads(analysis_file.read_text(encoding="utf-8"))
            results.append(ExampleSummary(
                id=d.name,
                title=data.get("caption", d.name)[:80],
                domain=data.get("domain", "unknown"),
                duration=data.get("duration", 0.0),
            ))
        return results

    def get_example(self, example_id: str) -> AnalysisResult | None:
        safe_id = Path(example_id).name
        analysis_file = self._dir / safe_id / "analysis.json"
        if not analysis_file.exists():
            return None
        data = json.loads(analysis_file.read_text(encoding="utf-8"))
        return AnalysisResult(**data)
