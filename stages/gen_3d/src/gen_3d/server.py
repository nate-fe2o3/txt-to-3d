"""FastAPI worker for stage 2 (image -> 3D, TRELLIS.2-4B).

Loads the pipeline once at startup and reuses it across requests.
Run with: uvicorn gen_3d.server:app --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
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


app = FastAPI(title="gen-3d worker", lifespan=lifespan)


class GenerateRequest(BaseModel):
    image_path: str
    seed: int
    out_path: str


class GenerateResponse(BaseModel):
    out_path: str


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    out = Path(req.out_path)
    run_inference(_pipeline, Path(req.image_path), req.seed, out)
    return GenerateResponse(out_path=str(out))


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": _pipeline is not None}
