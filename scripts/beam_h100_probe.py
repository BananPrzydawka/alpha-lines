# /// script
# requires-python = ">=3.12"
# dependencies = ["beam-client==0.2.211"]
# ///
"""Run via scripts/beam_h100_probe.sh to upload only this file."""
import json
import subprocess

from beam import Image, function


@function(
    name="alphalines-h100-probe",
    gpu="H100",
    gpu_count=1,
    cpu=1,
    memory="1Gi",
    image=Image(python_version="python3.12"),
    timeout=30,
    retries=0,
    headless=False,
)
def probe():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(devices) != 1 or "H100" not in devices[0]:
        raise RuntimeError(f"Expected one H100, received: {devices}")
    name, memory, driver = [field.strip() for field in devices[0].split(",")]
    return {"gpu": name, "memory_mib": int(memory), "driver": driver,
            "status": "H100 confirmed; no GPU workload executed"}


if __name__ == "__main__":
    print(json.dumps(probe.remote(), indent=2))
