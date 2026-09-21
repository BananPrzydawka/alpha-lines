"""Run the small inference benchmark on the project's configured Modal GPU."""
import json

import modal

from config import resources
from modal_image import image

app = modal.App("alphalines-model-speed")


@app.function(image=image, gpu=resources["gpu"], cpu=resources["cpus"],
              memory=resources["memory_gib"] * 1024, timeout=resources["timeout"])
def run_model(model: str, batch: int, calls: int, warmup: int):
    from model_speed.benchmark import benchmark

    return benchmark(model, batch, calls, warmup)


@app.local_entrypoint()
def main(model: str = "all", batch: int = 4096, calls: int = 32, warmup: int = 8):
    names = ("resnet", "maia", "katago")
    if model != "all" and model not in names:
        raise ValueError(f"model must be all or one of {names}")
    if min(batch, calls, warmup) < 1:
        raise ValueError("batch, calls, and warmup must be positive")
    results = [run_model.remote(name, batch, calls, warmup)
               for name in (names if model == "all" else [model])]
    print("\nModel       GPU ms/call   Wall ms/call")
    for row in results:
        print(f"{row['model']:10s} {row['ms_per_call']:11.3f} {row['wall_ms_per_call']:14.3f}")
    print(json.dumps(results, indent=2))
