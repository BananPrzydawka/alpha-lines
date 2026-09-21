"""Small CUDA inference benchmark; run with python -m model_speed.benchmark."""
import argparse
import gc
import json
import time

import torch

from config import settings
from models.katago import KataGoNet
from models.maia import MaiaNet
from models.resnet import ResNet
from scores.data import ScoreEncoder

MODELS = {"resnet": ResNet, "maia": MaiaNet, "katago": KataGoNet}


@torch.inference_mode()
def benchmark(model, batch=4096, calls=32, warmup=8):
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if min(batch, calls, warmup) < 1:
        raise ValueError("batch, calls, and warmup must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This benchmark requires a CUDA GPU with BF16 support")

    # Release compilation/graph state between models in a shared process.
    torch.compiler.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(1)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    net = MODELS[model]().eval().to(
        device="cuda", dtype=torch.bfloat16, memory_format=torch.channels_last
    )
    parameters = sum(p.numel() for p in net.parameters())
    # Synthetic board planes, encoded once. No Rust build, transfers, or encoding
    # in the timed region; all calls reuse the same resident batch.
    cells = torch.randint(4, (batch, 80), device="cuda", dtype=torch.uint8)
    inputs = ScoreEncoder().cuda()(cells).to(dtype=torch.bfloat16)
    args = (inputs,) if model == "maia" else (
        inputs, torch.randint(81, (batch, 2), device="cuda")
    )
    compiled = torch.compile(net, mode="max-autotune", fullgraph=True, dynamic=False)

    print(f"{model}: compiling and warming up ({warmup} calls)...", flush=True)
    torch.cuda.synchronize()
    warmup_started = time.perf_counter()
    for _ in range(warmup):
        compiled(*args)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - warmup_started

    # Time one block so synchronization and event recording do not interrupt
    # each call. max-autotune enables CUDA graphs where supported.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start.record()
    for _ in range(calls):
        output = compiled(*args)
        del output
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_started) * 1000 / calls
    ms = start.elapsed_time(end) / calls

    # Check the result outside timing without retaining CUDA graph outputs
    # across subsequent calls.
    output = compiled(*args)
    outputs = (output,) if model == "maia" else output
    expected = (batch, 2, 81) if model == "maia" else (batch, 10, 16)
    if any(value.shape != expected or not torch.isfinite(value).all().item()
           for value in outputs):
        raise RuntimeError(f"{model}: invalid model output")
    result = dict(model=model, parameters=parameters, batch=batch, calls=calls,
                  warmup=warmup, ms_per_call=ms, wall_ms_per_call=wall_ms,
                  boards_per_second=batch * 1000 / ms, compile_warmup_s=warmup_s,
                  gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  cuda=torch.version.cuda, dtype="bfloat16", mode="max-autotune",
                  config=settings)
    print(f"{model:7s}: {ms:.3f} ms/call | {wall_ms:.3f} wall ms/call | "
          f"{result['boards_per_second']:,.0f} boards/s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()
    names = MODELS if args.model == "all" else [args.model]
    results = [benchmark(name, args.batch, args.calls, args.warmup) for name in names]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
