"""Sweep native inference with CPU-resident zero planes, excluding game encoding."""
import json
import math
from pathlib import Path

import modal
from config import mcts, resources, settings
from modal_image import volume
from modal_mcts import benchmark_image, prepare_model, measure as measure_process

app = modal.App("alphalines-model-speed")


@app.function(image=benchmark_image, volumes={"/outputs": volume},
              gpu=resources["gpu"], cpu=resources["cpus"],
              memory=resources["memory_gib"] * 1024, timeout=resources["timeout"])
def measure(batch: int, seconds: float):
    import torch

    positions = batch // 2
    env, _, package, binary, gpu_name = prepare_model(positions, "model_speed", encoded_input=True)
    result = measure_process([
        str(binary), "--model", str(package), "--batch", str(positions),
        "--seconds", str(seconds),
    ], env)
    result.update({
        "gpu": gpu_name, "torch": torch.__version__,
        "heads": "policy-value", "dtype": "bfloat16",
        "backend": "rust-aoti-preencoded-zeros",
        "input_shape": [batch, 7, 10, 16], "input_transfer_bytes": batch * 7 * 10 * 16 * 2, "requested_seconds": seconds,
        "config": settings,
    })
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(batch: str = str(2 * mcts["b"]), seconds: float = 20,
         output: str = "logs/model-speed-preencoded/results.json"):
    """Sweep comma-separated batch sizes; seconds is the duration per batch size."""
    try:
        batches = [int(value.strip()) for value in batch.split(",")]
    except ValueError:
        raise ValueError("batch must be a comma-separated list of positive integers") from None
    if any(value < 2 or value % 2 for value in batches):
        raise ValueError("batch sizes count evals and must be positive even integers (2 evals per position)")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be positive and finite")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = {"runs": []}
    for size in batches:
        print(f"\nMeasuring batch={size}", flush=True)
        result = measure.remote(size, seconds)
        results["runs"].append(result)
        # Preserve completed measurements if a later batch fails.
        path.write_text(json.dumps(results, indent=2) + "\n")

    print("\nBatch (evals/call) | Evals/s | ms/call")
    for result in results["runs"]:
        print(f"{result['batch_evals']:17d} | {result['evals_per_s']:,.0f} | "
              f"{result['ms_per_call']:.3f}")
    print(f"Saved results to {path}")
