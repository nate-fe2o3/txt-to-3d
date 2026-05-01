import argparse
import logging
import os
import time
from pathlib import Path

# Env vars must be set before importing trellis2 / o_voxel.
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ATTN_BACKEND"] = "flash_attn_3"
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    str(Path(__file__).resolve().parent / "autotune_cache.json"),
)

import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/TRELLIS.2-4B"
PIPELINE_TYPE = "1024_cascade"
DECIMATION_TARGET = 150000
TEXTURE_SIZE = 1024


def generate(image_path: Path, seed: int, out_path: Path) -> None:
    logger.info("torch backend: device=cuda")
    logger.info("cuda device: name=%s", torch.cuda.get_device_name(0))

    logger.info("loading pipeline: model=%s", MODEL_ID)
    t0 = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
    pipeline.low_vram = False
    pipeline.cuda()
    logger.info("pipeline loaded in %.1fs", time.perf_counter() - t0)

    logger.info("loading image: %s", image_path)
    image = Image.open(image_path)

    logger.info("generating 3D: seed=%d pipeline_type=%s", seed, PIPELINE_TYPE)
    t1 = time.perf_counter()
    outputs, latents = pipeline.run(
        image,
        seed=seed,
        pipeline_type=PIPELINE_TYPE,
        return_latent=True,
    )
    logger.info("generation complete in %.1fs", time.perf_counter() - t1)

    mesh = outputs[0]
    mesh.simplify(16777216)

    logger.info("exporting GLB: decimation_target=%d texture_size=%d", DECIMATION_TARGET, TEXTURE_SIZE)
    t2 = time.perf_counter()
    grid_size = latents[2]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=grid_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=DECIMATION_TARGET,
        texture_size=TEXTURE_SIZE,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    glb.export(out_path, extension_webp=True)
    logger.info("GLB exported in %.1fs", time.perf_counter() - t2)
    logger.info("saved: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: image -> 3D model (TRELLIS.2-4B).")
    parser.add_argument("--image", type=Path, required=True, help="Input image.")
    parser.add_argument("--seed", type=int, required=True, help="RNG seed for reproducibility.")
    parser.add_argument("--out", type=Path, required=True, help="Output GLB path.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    generate(args.image, args.seed, args.out)


if __name__ == "__main__":
    main()
