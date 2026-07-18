"""
Probe v9: CUDA-13 base image (the test I skipped earlier).

Everything is CUDA 13 EXCEPT the base image, which was 12.1.1 -> so the
system CUPTI is libcupti.so.12. ncu counter path is NVPW -> CUPTI -> driver,
so a CUDA-12.1 CUPTI against a 580/CUDA-13 driver is a prime suspect for
"compatible driver library". This aligns the whole userspace to CUDA 13.

If the pull 404s, try: 13.0.0-devel-ubuntu24.04 or 13.0.1-devel-ubuntu22.04

Run:  uv run modal run ncu_probe.py
"""

import glob
import os
import subprocess
import sys

import modal

GPU = "L4"
NCU_PKG = "nsight-compute-2025.3.1"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install(NCU_PKG)
    .pip_install("torch", "numpy")
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all"})
)

app = modal.App("ncu-probe")


def _sh(label, cmd):
    print(f"\n----- {label} -----")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print((r.stdout + r.stderr).strip() or "(no output)")


@app.function(image=image, gpu=GPU, timeout=300)
def probe():
    _sh("system CUPTI version on path", "ldconfig -p | grep -i cupti || echo '(none)'")

    ncu = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))[-1]
    print(f"\nncu: {ncu}")

    workload = (
        "import torch;"
        "a=torch.randn(512,512,device='cuda');b=torch.randn(512,512,device='cuda');"
        "torch.cuda.synchronize();c=a@b;torch.cuda.synchronize()"
    )
    cmd = [
        ncu, "--set", "detailed", "--clock-control", "none",
        "--target-processes", "all", "--launch-count", "1",
        "--kernel-name", "regex:.*", "-f",
        sys.executable, "-c", workload,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout + r.stderr
    print(f"\n--- rc={r.returncode} ---")
    print(out)

    print("\n=== verdict ===")
    if r.returncode == 0 and ("Duration" in out or "sm__" in out or "gpu__" in out):
        print("OK: CUDA-13 base fixed it. The 12.1 CUPTI was the problem.")
    elif "LibraryNotLoaded" in out or "compatible driver" in out:
        print("Same error even fully on CUDA 13 => Modal/KVM PM-counter limitation. Pivot.")
    else:
        print("UNCLEAR -- inspect output.")