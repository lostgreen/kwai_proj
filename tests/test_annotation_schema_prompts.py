import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    REPO_ROOT
    / "video_proxy"
    / "data"
    / "pipelines"
    / "proxy_construction"
    / "annotation"
    / "annotation_schema.py"
)


def _load_schema_module():
    spec = importlib.util.spec_from_file_location("annotation_schema_under_test", SCHEMA_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_annotation_schema_only_exports_scene_first_prompt_builders():
    schema = _load_schema_module()

    active_prompt_builders = {
        "get_scene_first_prompt",
        "get_scene_first_l3_prompt",
        "get_l1_aggregation_prompt",
    }
    removed_prompt_builders = {
        "get_classification_prompt",
        "get_archetype_merged_prompt",
        "get_paradigm_merged_prompt",
        "get_unified_merged_prompt",
        "get_l2l3_first_prompt",
    }

    for name in active_prompt_builders:
        assert callable(getattr(schema, name))

    for name in removed_prompt_builders:
        assert not hasattr(schema, name)
