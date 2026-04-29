# Alpha PoC: Text → 3D Game Asset Pipeline

## Context

The user wants a tool they will use repeatedly to generate static and interactive prop assets (rocks, weapons) for a Bevy-based video game. This plan covers **Alpha (α)** only: a vertical-slice proof-of-concept that takes a text prompt, runs it through a two-stage generative pipeline, and produces a `.glb` on disk that can be viewed in a minimal Bevy scene.

**Decision criterion for α:** ≥3 of the 5 prompts in `prompts.txt` look usable as prototyping placeholders in a Bevy scene at gameplay-camera distance. Pass → invest in stage β (Mac↔GPU split, image-curation UX, caching). Fail → reconsider whether this pipeline is worth pursuing.

Out of scope for α: post-processing (decimation, retopo, delighting), caching, multi-user, Mac↔GPU split, anything resembling product polish.

## Architecture

Two-stage pipeline, run entirely on a single rented **A10 24GB** cloud GPU. The M40 is being skipped — Maxwell lacks flash-attn / bf16 / Tensor Cores, and the user's time is worth more than fighting it.

```
prompt --> [FLUX.2-klein-4B via diffusers] --> generated.png
                                                    |
                                                    v
                             [TRELLIS.2-4B (built-in BRIA-RMBG-2.0 bg removal)]
                                                    |
                                                    v
                                                model.glb
```

BiRefNet has been dropped — TRELLIS-2 ships its own background removal (BRIA-RMBG-2.0, with rembg fallback) in preprocessing. Reintroduce only if mask quality turns out to be the bottleneck.

## Repo Layout

Two **independent** uv projects (not a workspace — workspaces share deps, the opposite of what we want here, since FLUX and TRELLIS are likely to disagree on torch/transformers pins). The orchestrator shells out to each stage as a subprocess; each stage gets full 24GB on entry and full reclaim on process exit.

```
text-to-3d/                          # this repo (already exists, currently default uv init scaffold)
  pyproject.toml                     # orchestrator only — argparse, subprocess, pathlib. No ML deps.
  prompts.txt                        # already written
  src/orchestrator/
    main.py                          # CLI: --prompt, --seed; creates output dir; calls each stage
  stages/
    image_gen/
      pyproject.toml                 # diffusers, transformers, accelerate, torch, pillow
      src/image_gen/run.py           # CLI: --prompt --seed --out PATH
    gen_3d/
      pyproject.toml                 # trellis2 + its CUDA deps (flash-attn, xformers, torch_scatter, etc.)
      src/gen_3d/run.py              # CLI: --image PATH --out PATH
  outputs/                           # gitignored; one subdir per run
    YYYY-MM-DD_HH-MM-SS_<slug>/
      prompt.txt
      config.json                    # seed, model versions, params
      01_generated.png
      04_model.glb
      log.txt
```

Sibling repo (separate, not nested):

```
~/code/text-to-3d-viewer/            # Bevy α-min viewer — NEW repo, cargo init
  Cargo.toml
  src/main.rs                        # ~150 LOC
```

## Stage 1: image_gen

**Entry:** `uv run --project stages/image_gen python -m image_gen.run --prompt "..." --seed 42 --out outputs/.../01_generated.png`

**Implementation:**
- Use `diffusers.Flux2KleinPipeline` (user verified it exists; Apache 2.0).
- FP16, run on CUDA.
- Append a fixed suffix to every prompt to bias toward TRELLIS-friendly framing: `", single object centered, three-quarter view, plain neutral background, even studio lighting, no shadows"`. This is the only prompt-engineering for α; do not over-tune until you see baseline output.
- Generator seeded from `--seed`; seed also written to `config.json`.
- Output: 1024×1024 PNG.

## Stage 2: gen_3d

**Entry:** `uv run --project stages/gen_3d python -m gen_3d.run --image outputs/.../01_generated.png --out outputs/.../04_model.glb`

**Implementation:**
- Load `microsoft/TRELLIS.2-4B` per its README. The README's reference invocation is the source of truth — do not paraphrase it from memory.
- Let TRELLIS's own preprocessing handle background removal (BRIA-RMBG-2.0). Pass the RGB image as-is.
- Export GLB via TRELLIS's built-in mesh exporter.
- **Reference repo to skim before implementing:** [PRITHIVSAKTHIUR/TRELLIS.2-Text-to-3D](https://github.com/PRITHIVSAKTHIUR/TRELLIS.2-Text-to-3D) — does the same Z-Image+TRELLIS chain; the TRELLIS invocation pattern transfers directly.

## Orchestrator

**Entry:** `python -m orchestrator --prompt "a wooden barrel..." --seed 42`

Trivial responsibilities only:
1. Make `outputs/<timestamp>_<slug>/` (slug = first 4 words of prompt, sanitized).
2. Write `prompt.txt` and skeleton `config.json`.
3. `subprocess.run(["uv", "run", "--project", "stages/image_gen", ...])` — fail fast on non-zero exit.
4. Same for `stages/gen_3d`.
5. Tee combined stdout/stderr to `log.txt`.

No ML imports in this process. ~80 LOC.

## Bevy Viewer (sibling repo)

α-min scope only:
- `cargo run -- ../text-to-3d/outputs/<run>/04_model.glb` (or watches a dir; see below).
- `bevy_panorbit_camera` for orbit/zoom (one dep).
- One `DirectionalLight` at ~45° elevation, low ambient, neutral grey ground plane.
- 1m reference cube next to spawn point for scale sanity.
- Asset hot-reload enabled via `AssetPlugin { watch_for_changes_override: Some(true), .. }` so re-runs of the orchestrator update the scene without restarting the viewer.

Defer: wireframe toggle, lighting cycling, multi-asset comparison, HUD. Add only when the friction is felt during grading.

## Hardware / Runtime

- **A10 24GB rented from work cloud account.** Provision before starting — do not get stuck on cloud auth at midnight.
- HF cache: `export HF_HOME=/large-disk/hf-cache` before first run. Combined download is ~16–20 GB.
- Verify TRELLIS.2-4B HF gating status; `huggingface-cli login` if needed.

## Verification

α is "done" when both halves below pass:

**1. Pipeline runs end-to-end on all 5 prompts in `prompts.txt`.**
```
for line in prompts.txt:
    python -m orchestrator --prompt "<prompt>" --seed 42
```
Each run produces `04_model.glb`. No crashes, no OOMs.

**2. Grading pass in Bevy viewer.**
- Load each GLB in `text-to-3d-viewer`.
- For each: judge against the 1m reference cube under the default lighting whether it is recognizable and usable as a prototyping placeholder at gameplay camera distance.
- **Pass condition:** ≥3 of 5 judged usable. Record verdict in a `RESULTS.md` in the run directory.

If pipeline runs but <3/5 pass: alpha is informative but negative. Decide based on *which* prompts failed whether the issue is fixable (prompt-engineering, seed-rolling, image-gen tuning) or fundamental (TRELLIS just can't do hard-surface, etc.).

## Risks / Mitigations Already Acknowledged

- **TRELLIS-2 dependency setup will eat ~1 day.** Mitigation: independent uv project for stage 2 isolates the pain; if `uv lock` fights, fall through to `pip install -r requirements.txt` from the TRELLIS repo verbatim.
- **First TRELLIS run on A10 ≈ 1–3 min per asset.** Plan for ~30 min total to grade all 5 prompts.
- **TRELLIS bakes lighting into albedo.** Will look slightly off in Bevy vs. a glTF viewer with HDRI. Acceptable for prototyping bar (e); will need delighting for shipping bar (c) later.
- **TRELLIS GLBs may render oddly first try in Bevy** (face culling, normals, alpha mode). If a GLB looks fine in [gltf-viewer.donmccurdy.com](https://gltf-viewer.donmccurdy.com) but bad in Bevy, suspect Bevy material config before suspecting the model.

## Realistic Time Budget

- Day 1: Provision A10 + download weights + stand up `stages/image_gen` + first FLUX image.
- Day 2: Stand up `stages/gen_3d` + first end-to-end orchestrator run.
- Day 3: Bevy α-min viewer + grade 5 prompts + write `RESULTS.md`.
- Day 4: Slack for the unexpected.

Past day 4 with no verdict → stop and re-evaluate, do not grind.
