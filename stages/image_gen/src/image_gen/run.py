import argparse
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

FRAMING_SUFFIX = ", single object centered, three-quarter view, plain neutral background, even studio lighting, no shadows"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"


def generate(prompt: str, seed: int, out_path: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    full_prompt = prompt + FRAMING_SUFFIX
    generator = torch.Generator(device=device).manual_seed(seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = pipe(
        prompt=full_prompt,
        generator=generator,
        height=1024,
        width=1024,
    ).images[0]
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: text -> image (FLUX.2-klein-4B).")
    parser.add_argument("--prompt", required=True, help="Text prompt describing the object.")
    parser.add_argument("--seed", type=int, required=True, help="RNG seed for reproducibility.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    args = parser.parse_args()

    generate(args.prompt, args.seed, args.out)


if __name__ == "__main__":
    main()
