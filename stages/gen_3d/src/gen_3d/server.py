"""FastAPI worker for stage 2 (image -> 3D, TRELLIS.2-4B).

Loads the pipeline once at startup and reuses it across requests.
Run with: uvicorn gen_3d.server:app --host 0.0.0.0 --port 9002
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, FastAPI, Query
from fastapi.responses import Response

from .pipeline import load_pipeline, run_inference

logger = logging.getLogger(__name__)

# Module-level cache of the loaded pipeline. Populated by `lifespan` at startup.
_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _pipeline = load_pipeline()
    yield
    _pipeline = None


app = FastAPI(title="gen-3d worker", lifespan=lifespan)


@app.post("/generate")
def generate(
    image: Annotated[bytes, Body(media_type="application/octet-stream")],
    seed: Annotated[int, Query()],
) -> Response:
    glb_bytes = run_inference(_pipeline, image, seed)
    return Response(content=glb_bytes, media_type="model/gltf-binary")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": _pipeline is not None}
