import io
import logging
import time

import torch
from diffusers import Flux2KleinPipeline

logger = logging.getLogger(__name__)

FRAMING_SUFFIX = ", single object centered, three-quarter view, plain neutral background, even studio lighting, no shadows"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def load_pipeline() -> Flux2KleinPipeline:
    """Load FLUX.2-klein-4B with CPU offload so it can share a GPU with the gen_3d worker.

    Without offload, FLUX.2-klein holds ~20 GB on GPU (transformer + Qwen3 text encoder +
    VAE all resident). With offload, only the active component lives on GPU at any moment,
    leaving headroom for TRELLIS. Costs ~5-15s per request for CPU<->GPU transfers.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    logger.info("torch backend: device=%s dtype=%s", device, dtype)
    if device == "cuda":
        logger.info("cuda device: name=%s", torch.cuda.get_device_name(0))

    logger.info("loading pipeline: model=%s", MODEL_ID)
    t0 = time.perf_counter()
    pipeline = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    if device == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    logger.info("pipeline loaded in %.1fs", time.perf_counter() - t0)
    return pipeline


def run_inference(pipeline: Flux2KleinPipeline, prompt: str, seed: int) -> bytes:
    """Generate one image with a pre-loaded pipeline. Returns PNG bytes."""
    full_prompt = prompt + FRAMING_SUFFIX
    # With cpu_offload, pipeline.device reports CPU (resident location). For the generator
    # we want the execution device — the one where computation actually happens.
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    logger.info("generating: seed=%d resolution=1024x1024 prompt=%r", seed, full_prompt)

    t0 = time.perf_counter()
    image = pipeline(
        prompt=full_prompt,
        generator=generator,
        height=1024,
        width=1024,
    ).images[0]
    logger.info("generation complete in %.1fs", time.perf_counter() - t0)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    logger.info("encoded PNG: %d bytes", len(png_bytes))
    return png_bytes
