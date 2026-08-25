"""Time the CUDA kernels in src/ on a Modal GPU.  uv run modal run cuda/modal_cuda.py

No profiling: Modal blocks the performance counters. Timing only.
Compilation happens remotely because Arch-built binaries won't load on Ubuntu.
"""

import os
import subprocess
from pathlib import Path

import modal

ARCH = "sm_90a"
GPU = "H100"
ITERS = 200
IMAGE = "nvidia/cuda:13.3.1-devel-ubuntu24.04"

app = modal.App("alphalines-cuda")
image = (
    modal.Image.from_registry(IMAGE, add_python="3.12")
    .apt_install("make")
    .add_local_dir(Path(__file__).parent, remote_path="/root/cuda",
                   ignore=["build/**", "out/**"])
)


@app.function(image=image, gpu=GPU, timeout=600)
def bench():
    subprocess.run("nvidia-smi", shell=True)
    subprocess.run(f"make bins ARCH={ARCH}", shell=True, cwd="/root/cuda", check=True)
    bld = Path("/root/cuda/build") / ARCH
    for b in sorted(bld.iterdir()):
        if b.is_file() and os.access(b, os.X_OK) and "." not in b.name:
            print(f"\n== {b.name}", flush=True)
            subprocess.run([str(b), str(ITERS)])


@app.local_entrypoint()
def main():
    bench.remote()