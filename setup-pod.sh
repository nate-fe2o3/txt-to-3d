#!/usr/bin/env bash
# One-time pod setup: install OS-level deps that aren't preserved across RunPod
# container restarts. Run this once per fresh pod, before `uv sync` in any stage.
#
# Specifically: Triton (used by flex_gemm in gen_3d) JIT-compiles C extensions at
# import time and needs Python's C headers. These ship in python3.10-dev which is
# not preinstalled on the standard PyTorch RunPod template.
#
# Idempotent — apt-get install is a no-op if already present.

set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get not found — this script targets Debian/Ubuntu containers." >&2
    exit 1
fi

apt-get update
apt-get install -y \
    python3.10-dev \
    gcc \
    tmux

echo
echo "Done. OS-level deps installed."
echo "Next: bash setup-venvs.sh, then uv sync in each stage."
