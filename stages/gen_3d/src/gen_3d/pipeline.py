import io
import logging
import os
import tempfile
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


def load_pipeline() -> Trellis2ImageTo3DPipeline:
    """Load TRELLIS.2-4B onto GPU with CPU offload. Slow: ~5min cold (NFS reads + deserialize)."""
    logger.info("torch backend: device=cuda")
    logger.info("cuda device: name=%s", torch.cuda.get_device_name(0))

    logger.info("loading pipeline: model=%s", MODEL_ID)
    t0 = time.perf_counter()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
    # 24 GB cards (A5000/A10/4090) need CPU offload to avoid OOM during mesh
    # post-processing. Adds ~30-60s to total runtime.
    pipeline.low_vram = True
    pipeline.cuda()
    logger.info("pipeline loaded in %.1fs", time.perf_counter() - t0)
    return pipeline


def run_inference(
    pipeline: Trellis2ImageTo3DPipeline,
    image_bytes: bytes,
    seed: int,
) -> bytes:
    """Generate one GLB with a pre-loaded pipeline. Returns GLB bytes."""
    # TRELLIS-2 handles bg removal itself, so RGB is sufficient. Forcing the mode
    # normalizes RGBA / palette / 16-bit inputs into the one form the pipeline accepts.
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    logger.info("loaded input image: %dx%d mode=%s", image.width, image.height, image.mode)

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

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        tmp_path = Path(f.name)
    try:
        glb.export(tmp_path, extension_webp=True)
        glb_bytes = tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
    logger.info("GLB exported in %.1fs (%d bytes)", time.perf_counter() - t2, len(glb_bytes))
    return glb_bytes
