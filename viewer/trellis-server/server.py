"""
TRELLIS API Server
Scene Composer backend - image/text -> 3D Gaussian Splatting generation

Run:
  # Mock mode (no GPU):
  TRELLIS_MOCK=true uvicorn server:app --host 0.0.0.0 --port 8000

  # Real TRELLIS mode:
  uvicorn server:app --host 0.0.0.0 --port 8000

  # Or directly:
  TRELLIS_MOCK=true python3 server.py
"""
import asyncio
import base64
import time
import logging
import os
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from trellis_wrapper import TrellisGenerator
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TRELLIS API", version="1.0.0")

# CORS (allow Scene Composer frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response models ---

class TrellisParams(BaseModel):
    sparse_structure_steps: int = 12
    sparse_structure_cfg_strength: float = 7.5
    slat_steps: int = 12
    slat_cfg_strength: float = 3.0
    output_format: str = "splat"


class GenerateRequest(BaseModel):
    prompt_type: Literal["text", "image"] = "image"
    prompt_text: Optional[str] = None
    prompt_image: Optional[str] = None  # base64 encoded
    seed: int = -1
    params: TrellisParams = TrellisParams()


class GenerateResponse(BaseModel):
    status: str
    ply_data: Optional[str] = None       # base64 encoded
    gaussian_count: int = 0
    generation_time: float = 0.0
    format: str = "splat"
    thumbnail: Optional[str] = None
    error: Optional[str] = None


# --- Global state ---

generator: Optional[TrellisGenerator] = None
generation_lock = asyncio.Lock()


# --- Endpoints ---

@app.on_event("startup")
async def startup():
    global generator
    logger.info("Loading TRELLIS model...")
    generator = TrellisGenerator(settings)
    logger.info(f"TRELLIS ready (mock={generator.is_mock})")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": generator is not None,
        "mock_mode": generator.is_mock if generator else True,
        "gpu_available": generator.check_gpu() if generator else False,
    }


@app.get("/status")
async def status():
    return {"is_busy": generation_lock.locked()}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if request.prompt_type == "text" and not request.prompt_text:
        raise HTTPException(400, "prompt_text required for text mode")
    if request.prompt_type == "image" and not request.prompt_image:
        raise HTTPException(400, "prompt_image required for image mode")

    async with generation_lock:
        try:
            start_time = time.time()

            if request.prompt_type == "text":
                result = await asyncio.to_thread(
                    generator.generate_from_text,
                    request.prompt_text,
                    seed=request.seed,
                    params=request.params.model_dump(),
                )
            else:
                image_bytes = base64.b64decode(request.prompt_image)
                result = await asyncio.to_thread(
                    generator.generate_from_image,
                    image_bytes,
                    seed=request.seed,
                    params=request.params.model_dump(),
                )

            generation_time = time.time() - start_time
            data_base64 = base64.b64encode(result["ply_bytes"]).decode()

            return GenerateResponse(
                status="success",
                ply_data=data_base64,
                gaussian_count=result.get("gaussian_count", 0),
                generation_time=generation_time,
                format=result.get("format", "splat"),
                thumbnail=result.get("thumbnail_base64"),
            )
        except NotImplementedError as e:
            return GenerateResponse(status="error", error=str(e))
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return GenerateResponse(status="error", error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
