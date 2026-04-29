import argparse
import logging
import time
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

logger = logging.getLogger(__name__)

FRAMING_SUFFIX = ", single object centered, three-quarter view, plain neutral background, even studio lighting, no shadows"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def generate(prompt: str, seed: int, out_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("torch backend: device=%s dtype=%s", device, dtype)
    if device == "cuda":
        logger.info("cuda device: name=%s", torch.cuda.get_device_name(0))

    logger.info("loading pipeline: model=%s", MODEL_ID)
    t0 = time.perf_counter()
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)
    logger.info("pipeline loaded in %.1fs", time.perf_counter() - t0)

    full_prompt = prompt + FRAMING_SUFFIX
    generator = torch.Generator(device=device).manual_seed(seed)
    logger.info("generating: seed=%d resolution=1024x1024 prompt=%r", seed, full_prompt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.perf_counter()
    image = pipe(
        prompt=full_prompt,
        generator=generator,
        height=1024,
        width=1024,
    ).images[0]
    logger.info("generation complete in %.1fs", time.perf_counter() - t1)

    image.save(out_path)
    logger.info("saved: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: text -> image (FLUX.2-klein-4B).")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the object.")
    parser.add_argument("--seed", type=int, required=True, help="RNG seed for reproducibility.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    generate(args.prompt, args.seed, args.out)


if __name__ == "__main__":
    main()
