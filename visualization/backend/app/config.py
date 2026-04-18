from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIZ_")

    examples_dir: Path = Path(__file__).resolve().parent.parent.parent / "data" / "examples"
    model_name: str = "Qwen3-VL-4B-Instruct (RL fine-tuned)"
    vllm_url: str = ""
