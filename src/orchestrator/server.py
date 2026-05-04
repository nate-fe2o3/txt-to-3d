"""FastAPI gateway: single API surface that proxies to per-stage workers.

Bytes in, bytes out — no disk involvement. Workers stream PNG / GLB bytes
back; this gateway forwards them through to the client.

Run with: uvicorn orchestrator.server:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Worker URLs. Override via env vars on hosts where defaults conflict (e.g. RunPod's
# nginx template binds 8000/8001).
IMAGE_GEN_URL = os.environ.get("IMAGE_GEN_URL", "http://localhost:9001")
GEN_3D_URL = os.environ.get("GEN_3D_URL", "http://localhost:9002")

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
    _client = httpx.AsyncClient()
    yield
    await _client.aclose()
    _client = None


app = FastAPI(title="text-to-3d gateway", lifespan=lifespan)


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)


class HealthResponse(BaseModel):
    gateway: bool
    image_gen: bool
    gen_3d: bool


# --- worker call helpers ---


async def _call_image_worker(prompt: str, seed: int) -> bytes:
    """POST to image_gen worker, return PNG bytes. Raises HTTPException on failure."""
    try:
        resp = await _client.post(
            f"{IMAGE_GEN_URL}/generate",
            json={"prompt": prompt, "seed": seed},
            timeout=IMAGE_GEN_TIMEOUT,
        )
    except httpx.ConnectError as e:
        raise HTTPException(status_code=503, detail=f"image_gen worker unreachable: {e}") from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail=f"image_gen worker timed out after {IMAGE_GEN_TIMEOUT}s") from e

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"image_gen worker {resp.status_code}: {resp.text}",
        )
    return resp.content


async def _call_3d_worker(image_bytes: bytes, seed: int) -> bytes:
    """POST to gen_3d worker, return GLB bytes. Raises HTTPException on failure."""
    try:
        resp = await _client.post(
            f"{GEN_3D_URL}/generate",
            params={"seed": seed},
            content=image_bytes,
            headers={"Content-Type": "application/octet-stream"},
            timeout=GEN_3D_TIMEOUT,
        )
    except httpx.ConnectError as e:
        raise HTTPException(status_code=503, detail=f"gen_3d worker unreachable: {e}") from e
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail=f"gen_3d worker timed out after {GEN_3D_TIMEOUT}s") from e

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"gen_3d worker {resp.status_code}: {resp.text}",
        )
    return resp.content


# --- endpoints ---


@app.post("/image")
async def generate_image(req: PromptRequest) -> Response:
    seed = random.randint(0, SEED_MAX)
    png_bytes = await _call_image_worker(req.prompt, seed)
    return Response(content=png_bytes, media_type="image/png", headers={"X-Seed": str(seed)})


@app.post("/3d")
async def generate_3d(image: Annotated[UploadFile, File()]) -> Response:
    seed = random.randint(0, SEED_MAX)
    image_bytes = await image.read()
    glb_bytes = await _call_3d_worker(image_bytes, seed)
    return Response(content=glb_bytes, media_type="model/gltf-binary", headers={"X-Seed": str(seed)})


@app.post("/text-to-3d")
async def text_to_3d(req: PromptRequest) -> Response:
    """Full pipeline: prompt -> image -> GLB. Returns GLB bytes only.

    If the 3D step fails, the intermediate image is lost — the client must retry
    the whole pipeline. To keep the image (e.g. for picking among candidates),
    use /image and /3d separately.
    """
    image_seed = random.randint(0, SEED_MAX)
    threed_seed = random.randint(0, SEED_MAX)
    image_bytes = await _call_image_worker(req.prompt, image_seed)
    glb_bytes = await _call_3d_worker(image_bytes, threed_seed)
    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={"X-Image-Seed": str(image_seed), "X-3d-Seed": str(threed_seed)},
    )


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
