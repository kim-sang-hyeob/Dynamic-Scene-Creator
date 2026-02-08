import glob
import os
from pydantic_settings import BaseSettings

def _find_cached_model():
    """Find locally cached TRELLIS model snapshot path, fallback to HF repo name."""
    pattern = os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/*/pipeline.json")
    matches = glob.glob(pattern)
    if matches:
        return os.path.dirname(matches[0])
    return "microsoft/TRELLIS-image-large"

class Settings(BaseSettings):
    model_name: str = _find_cached_model()
    device: str = "cuda"
    max_vram_gb: float = 10.0
    port: int = 8000
    host: str = "0.0.0.0"
    worldlabs_api_key: str = ""

    class Config:
        env_prefix = "TRELLIS_"

settings = Settings()
