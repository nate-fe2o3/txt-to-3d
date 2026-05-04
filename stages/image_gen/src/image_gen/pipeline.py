import logging
import time
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

logger = logging.getLogger(__name__)

FRAMING_SUFFIX = ", single object centered, three-quarter view, plain neutral background, even studio lighting, no shadows"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def load_pipeline() -> Flux2KleinPipeline:
    """Load FLUX.2-klein-4B onto the available device. Slow: ~30s + weight download on first call."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("torch backend: device=%s dtype=%s", device, dtype)
    if device == "cuda":
        logger.info("cuda device: name=%s", torch.cuda.get_device_name(0))

    logger.info("loading pipeline: model=%s", MODEL_ID)
    t0 = time.perf_counter()
    pipeline = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    logger.info("pipeline loaded in %.1fs", time.perf_counter() - t0)
    return pipeline


def run_inference(pipeline: Flux2KleinPipeline, prompt: str, seed: int, out_path: Path) -> None:
    """Generate one image with a pre-loaded pipeline."""
    full_prompt = prompt + FRAMING_SUFFIX
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    logger.info("generating: seed=%d resolution=1024x1024 prompt=%r", seed, full_prompt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    image = pipeline(
        prompt=full_prompt,
        generator=generator,
        height=1024,
        width=1024,
    ).images[0]
    logger.info("generation complete in %.1fs", time.perf_counter() - t0)

    image.save(out_path)
    logger.info("saved: %s", out_path)
