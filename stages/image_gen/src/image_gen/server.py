"""FastAPI worker for stage 1 (text -> image, FLUX.2-klein-4B).

Loads the pipeline once at startup and reuses it across requests.
Run with: uvicorn image_gen.server:app --host 0.0.0.0 --port 9001
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

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


app = FastAPI(title="image-gen worker", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    seed: int


@app.post("/generate")
def generate(req: GenerateRequest) -> Response:
    png_bytes = run_inference(_pipeline, req.prompt, req.seed)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": _pipeline is not None}
