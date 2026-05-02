"""FastAPI gateway: single API surface that proxies to per-stage workers.

Run with: uvicorn orchestrator.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import random
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Worker URLs. Both workers run on localhost on the pod.
IMAGE_GEN_URL = "http://localhost:8001"
GEN_3D_URL = "http://localhost:8002"

# Where workers write artifacts. Gateway reads bytes back from here.
OUTPUTS_DIR = Path("/workspace/outputs")

# Per-endpoint timeouts when calling workers.
IMAGE_GEN_TIMEOUT = 120.0
GEN_3D_TIMEOUT = 600.0

# Random seed range — fits in a 32-bit signed int (matches torch.Generator).
SEED_MAX = 2**31 - 1

_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _client = httpx.AsyncClient()
    yield
    await _client.aclose()
    _client = None


app = FastAPI(title="text-to-3d gateway", lifespan=lifespan)


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1)


class HealthResponse(BaseModel):
    gateway: bool
    image_gen: bool
    gen_3d: bool


@app.post("/image")
async def generate_image(req: ImageRequest) -> Response:
    seed = random.randint(0, SEED_MAX)
    out_path = OUTPUTS_DIR / f"{uuid.uuid4()}.png"

    try:
        worker_resp = await _client.post(
            f"{IMAGE_GEN_URL}/generate",
            json={"prompt": req.prompt, "seed": seed, "out_path": str(out_path)},
            timeout=IMAGE_GEN_TIMEOUT,
        )
    except httpx.ConnectError as e:
        raise HTTPException(status_code=503, detail=f"image_gen worker unreachable: {e}") from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail=f"image_gen worker timed out after {IMAGE_GEN_TIMEOUT}s") from e

    if worker_resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"image_gen worker error: {worker_resp.text}")

    png_bytes = out_path.read_bytes()
    return Response(content=png_bytes, media_type="image/png", headers={"X-Seed": str(seed)})


@app.post("/3d")
async def generate_3d(image: Annotated[UploadFile, File()]) -> Response:
    seed = random.randint(0, SEED_MAX)
    upload_path = OUTPUTS_DIR / f"{uuid.uuid4()}_input.png"
    out_path = OUTPUTS_DIR / f"{uuid.uuid4()}.glb"

    upload_bytes = await image.read()
    upload_path.write_bytes(upload_bytes)

    try:
        worker_resp = await _client.post(
            f"{GEN_3D_URL}/generate",
            json={"image_path": str(upload_path), "seed": seed, "out_path": str(out_path)},
            timeout=GEN_3D_TIMEOUT,
        )
    except httpx.ConnectError as e:
        raise HTTPException(status_code=503, detail=f"gen_3d worker unreachable: {e}") from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail=f"gen_3d worker timed out after {GEN_3D_TIMEOUT}s") from e

    if worker_resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"gen_3d worker error: {worker_resp.text}")

    glb_bytes = out_path.read_bytes()
    return Response(content=glb_bytes, media_type="model/gltf-binary", headers={"X-Seed": str(seed)})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    async def _check(url: str) -> bool:
        try:
            r = await _client.get(f"{url}/health", timeout=5.0)
            return r.status_code == 200 and r.json().get("ok", False)
        except Exception:
            return False

    return HealthResponse(
        gateway=True,
        image_gen=await _check(IMAGE_GEN_URL),
        gen_3d=await _check(GEN_3D_URL),
    )
