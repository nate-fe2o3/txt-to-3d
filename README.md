# text-to-3d

Text-to-3D game-asset pipeline. A prompt goes in, a `.glb` comes out.

The pipeline is two generative models chained together:
1. **FLUX.2-klein-4B** (`black-forest-labs/FLUX.2-klein-4B`) generates an image from the prompt.
2. **TRELLIS.2-4B** (`microsoft/TRELLIS.2-4B`) turns that image into a textured 3D mesh.

Each model runs in its own FastAPI worker process (different Python versions and conflicting deps make a single process impossible). A small gateway service exposes a unified API on top.

```
client → gateway (:9000) → image_gen worker (:9001)  ← FLUX.2-klein-4B
                         → gen_3d worker     (:9002) ← TRELLIS.2-4B
```

---

## Requirements

| | Minimum |
|---|---|
| GPU | 24 GB VRAM, Ampere or newer (A10, A5000, 4090, A100). Older Maxwell/Pascal/Volta cards will not work — TRELLIS depends on `flash-attn-3` (Ampere+). |
| OS | Linux x86_64. The TRELLIS-2 prebuilt CUDA wheels are `linux_x86_64` only. |
| CUDA driver | 12.4+. Older drivers crash at torch import. |
| Disk | ~50 GB free for model weights (FLUX ≈ 8 GB, TRELLIS ≈ 16 GB, plus venvs and caches). |
| Python | 3.10 for `gen_3d` (hard requirement — TRELLIS wheels are `cp310`); 3.12+ for `image_gen` and the gateway. |
| HuggingFace account | Required. TRELLIS-2 uses the **gated** `facebook/dinov3-vitl16-pretrain-lvd1689m` model — you must accept its license on HuggingFace and authenticate. |

---

## Setup

### 1. Get the code

```bash
# This repo:
git clone <this-repo-url> ~/text-to-3d

# TRELLIS-2 source — not pip-installable, must be cloned and added to PYTHONPATH:
git clone https://github.com/microsoft/TRELLIS.2.git ~/trellis2-src
```

Adjust paths as you like. The README assumes `~/text-to-3d` and `~/trellis2-src`. If you change them, update the env vars in step 6.

### 2. Install OS-level dependencies

The gen_3d worker uses Triton, which JIT-compiles a CUDA driver helper in C at first import. That requires the Python development headers and a C compiler.

```bash
sudo apt-get update
sudo apt-get install -y python3.10-dev gcc
```

On RunPod containers these are not preinstalled and they don't survive a pod restart, so you'll re-run this on every fresh pod.

### 3. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Add the `export PATH=...` line to your shell rc so it persists.

### 4. HuggingFace authentication

TRELLIS-2 pulls Meta's DINOv3 ViT-L/16 as its image-conditioning encoder, which is a **gated** model. Two steps:

1. Visit <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m> while logged in to your HF account, click **Agree and access repository**, accept the license. Approval is usually instant but can be up to ~24h.

2. Create a read-scope token at <https://huggingface.co/settings/tokens>, then on the machine:

   ```bash
   huggingface-cli login   # paste token when prompted
   # OR set the env var directly:
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

   Verify: `huggingface-cli whoami` should print your username.

### 5. Configure environment

Add to your shell rc (`~/.bashrc` or equivalent):

```bash
# Where HuggingFace caches model weights (~24 GB total). Point this at persistent
# storage if you're on a host where the home directory is ephemeral.
export HF_HOME=$HOME/hf-cache

# TRELLIS-2 isn't pip-installable, so we add its source to PYTHONPATH.
export PYTHONPATH=$HOME/trellis2-src:$PYTHONPATH

# HF token (if not using `huggingface-cli login`).
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`source ~/.bashrc` (or open a new shell).

### 6. Install Python dependencies

Three independent uv projects: gateway at the repo root, plus a stage worker each in `stages/image_gen/` and `stages/gen_3d/`. Each has its own `pyproject.toml` and its own venv.

```bash
cd ~/text-to-3d                  && uv sync
cd ~/text-to-3d/stages/image_gen && uv sync
cd ~/text-to-3d/stages/gen_3d    && uv sync   # this one is heavy: ~10–20 min, big CUDA wheels
```

The gen_3d sync downloads several large prebuilt wheels (`cumesh`, `flex_gemm`, `o_voxel`, `nvdiffrast`, `flash-attn-3`) directly from JeffreyXiang's GitHub releases. No source compilation, but the download is slow.

---

## Running the servers

Three processes, each on its own port. They each take a while to load model weights on first start (image_gen ~1–2 min, gen_3d ~5–10 min cold). Start them however you like — separate terminals, `screen`, `nohup`, systemd, whatever.

```bash
# Worker 1: image generation
cd ~/text-to-3d/stages/image_gen
uv run uvicorn image_gen.server:app --host 0.0.0.0 --port 9001

# Worker 2: 3D generation (in a separate terminal)
cd ~/text-to-3d/stages/gen_3d
uv run uvicorn gen_3d.server:app --host 0.0.0.0 --port 9002

# Gateway (in a third terminal)
cd ~/text-to-3d
uv run uvicorn orchestrator.server:app --host 0.0.0.0 --port 9000
```

The gateway expects workers at `localhost:9001` and `localhost:9002` by default. Override with `IMAGE_GEN_URL` and `GEN_3D_URL` env vars if needed.

Verify everything is up:

```bash
curl http://localhost:9000/health
# {"gateway":true,"image_gen":true,"gen_3d":true}
```

---

## API

All endpoints live on the gateway (`:9000`). Workers are internal — clients should not call them directly.

| Method | Path | Body | Returns | Headers |
|---|---|---|---|---|
| `POST` | `/image` | `{"prompt": str}` | `image/png` bytes | `X-Seed: <int>` |
| `POST` | `/3d` | `multipart/form-data` with `image` field (PNG bytes) | `model/gltf-binary` bytes | `X-Seed: <int>` |
| `POST` | `/text-to-3d` | `{"prompt": str}` | `model/gltf-binary` bytes | `X-Image-Seed`, `X-3d-Seed` |
| `GET` | `/health` | — | `{"gateway": bool, "image_gen": bool, "gen_3d": bool}` | — |

Seeds are random per request. They're returned in headers for traceability but cannot be supplied as input.

OpenAPI / Swagger UI: `http://localhost:9000/docs`.

### Examples

**Generate three image candidates, pick one, get a 3D model from it:**

```bash
for i in 1 2 3; do
  curl -X POST http://localhost:9000/image \
       -H 'Content-Type: application/json' \
       -d '{"prompt":"a wooden barrel"}' \
       --output candidate_$i.png
done
# review candidate_1.png, candidate_2.png, candidate_3.png — pick one
curl -X POST http://localhost:9000/3d \
     -F image=@candidate_2.png \
     --output barrel.glb
```

**Single-shot full pipeline (no review step):**

```bash
curl -X POST http://localhost:9000/text-to-3d \
     -H 'Content-Type: application/json' \
     -d '{"prompt":"a wooden barrel"}' \
     --output barrel.glb
```

---

## Repo layout

```
text-to-3d/
  pyproject.toml                       # gateway deps (fastapi, httpx, uvicorn, pydantic)
  src/orchestrator/
    server.py                          # gateway: 3 endpoints + /health
  stages/
    image_gen/
      pyproject.toml                   # FLUX.2-klein deps (diffusers, transformers, torch)
      src/image_gen/
        pipeline.py                    # load_pipeline() + run_inference()
        server.py                      # FastAPI worker
    gen_3d/
      pyproject.toml                   # TRELLIS-2 deps (torch 2.6 + cu124, prebuilt CUDA wheels)
      src/gen_3d/
        pipeline.py                    # load_pipeline() + run_inference()
        server.py                      # FastAPI worker
  prompts.txt                          # alpha test prompt set
  PLAN.md                              # original architecture plan
```

The two stage projects deliberately have separate `pyproject.toml`s and venvs because their pinned dependencies don't reconcile (different Python versions, different torch versions, different transformers versions).

---

## Troubleshooting

**`Could not import module 'Qwen3ForCausalLM'` / `PreTrainedModel`** at image_gen startup
→ The CUDA driver is too old for the installed torch version. Update the driver or downgrade torch. See the warning text in the stack trace for the driver version your system reports.

**`Cannot access gated repo for url ... facebook/dinov3-vitl16-pretrain-lvd1689m`**
→ You haven't accepted the DINOv3 license, or you haven't authenticated. See setup step 4.

**`Python.h: No such file or directory`** when starting gen_3d
→ Missing Python development headers. Run `apt-get install -y python3.10-dev gcc`.

**`CUDA out of memory` during the shape SLat sampler in gen_3d**
→ Another process is holding GPU memory. `nvidia-smi` to find it, `kill -9 <PID>`, restart gen_3d. A common cause is a stale uvicorn from a previous run.

**`address already in use`** on port 9001/9002/9000
→ Old uvicorn still running. `fuser -k 9001/tcp` (or 9002/9000), then restart.

**`subprocess.CalledProcessError` from `flex_gemm` autotuner**
→ Same as the `Python.h` issue: install `python3.10-dev`. Triton compiles C extensions at first use and needs the headers.

**RunPod-specific: NFS `Stale file handle (os error 116)` during `uv sync`**
→ The project source is on an NFS mount; uv's atomic-rename operations don't work there. Move the venv off NFS by setting `UV_PROJECT_ENVIRONMENT=/root/venvs/<stage-name>` before running `uv sync`, or relocate the project source to a non-NFS path like `/local_disk0`.
