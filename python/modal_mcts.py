"""Run independent Rust MCTS threads on one Modal GPU and sweep worker counts."""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import urllib.request

import modal
from config import mcts, benchmark as defaults, resources, settings
from modal_image import image, volume, project_dir

# Rust sources and the shared config are mounted at runtime.
source = project_dir / "rust"
benchmark_image = image.add_local_dir(
    source, remote_path="/root/rust",
    ignore=["target/", "tests/", "benchmarks/", "xcheck/", "__pycache__/"],
)
app = modal.App("alphalines-mcts-speed")


def measure(command, env):
    active, stop = threading.Event(), threading.Event()
    samples = []

    def monitor():
        while not stop.wait(1):
            if not active.is_set():
                continue
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                util, memory = result.stdout.strip().splitlines()[0].split(",")
                samples.append((float(util), float(memory)))

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    result = None
    try:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("BENCHMARK_START"):
                active.set()
            if line.startswith("RESULT "):
                active.clear()
                result = json.loads(line.removeprefix("RESULT "))
        if process.wait() != 0 or result is None:
            raise RuntimeError("Rust benchmark failed; see subprocess output")
    finally:
        stop.set()
        watcher.join()
    if samples:
        result["gpu_utilization_mean_pct"] = statistics.mean(x[0] for x in samples)
        result["gpu_memory_peak_mib"] = max(x[1] for x in samples)
    return result


def rust_source_key(root: Path) -> str:
    """Identify source contents independently of mounted file timestamps."""
    rust = root / "rust"
    paths = [rust / "Cargo.toml", rust / "Cargo.lock", rust / "build.rs", root / "config.json"]
    for directory in ("src", "examples", "native"):
        paths.extend(path for path in (rust / directory).rglob("*") if path.is_file())
    key = hashlib.sha256()
    for path in sorted(paths):
        key.update(str(path.relative_to(root)).encode() + b"\0")
        key.update(path.read_bytes() + b"\0")
    return key.hexdigest()[:20]


def prepare_model(batch: int, example: str, encoded_input: bool = False):
    """Build a native benchmark and reuse the shared MCTS export cache."""
    import torch

    root = Path("/root")
    cache = Path("/outputs/mcts-bench")
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHON=sys.executable, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               CARGO_HOME=str(cache / "cargo"), RUSTUP_HOME=str(cache / "rustup"),
               # Runtime mounts can retain timestamps older than cached Cargo outputs.
               # Separate builds by contents so an old binary cannot appear fresh.
               CARGO_TARGET_DIR=str(cache / "target" / rust_source_key(root)),
               TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"))
    cargo = Path(env["CARGO_HOME"]) / "bin/cargo"
    if not cargo.exists():
        installer = Path("/tmp/mcts-rustup.sh")
        urllib.request.urlretrieve("https://sh.rustup.rs", installer)
        subprocess.run(["sh", str(installer), "-y", "--profile", "minimal", "--no-modify-path"], env=env, check=True)
    env["PATH"] = f"{cargo.parent}:{env['PATH']}"
    subprocess.run([str(cargo), "build", "--manifest-path", str(root / "rust/Cargo.toml"),
                    "--release", "--features", "compiled-model", "--example", example], env=env, check=True)

    gpu_name = torch.cuda.get_device_name(0)
    key = hashlib.sha256()
    for name in ("python/export_mcts.py", "python/model.py", "python/config.py", "config.json"):
        key.update((root / name).read_bytes())
    key.update(f"{torch.__version__}:{gpu_name}:{torch.cuda.get_device_capability(0)}:{batch}".encode())
    if encoded_input:
        key.update(b"encoded-input-v1")
    package = cache / f"model-{key.hexdigest()[:20]}.pt2"
    if not package.exists() or not Path(str(package) + ".meta").exists():
        subprocess.run([sys.executable, str(root / "python/export_mcts.py"), "--batch", str(batch),
                        "--output", str(package)] + (["--encoded-input"] if encoded_input else []), env=env, check=True)
    volume.commit()
    binary = Path(env["CARGO_TARGET_DIR"]) / "release/examples" / example
    return env, cargo, package, binary, gpu_name


@app.function(image=benchmark_image, volumes={"/outputs": volume}, gpu=resources["gpu"],
              cpu=resources["cpus"], memory=resources["memory_gib"] * 1024,
              timeout=resources["timeout"])
def benchmark(counts: list[int], seconds: float, repeats: int, warmup_steps: int,
              variant: str, games: int, batch: int, threshold: int, sims: int):
    import torch

    env, cargo, package, binary, gpu_name = prepare_model(batch, "speed", encoded_input=True)
    cache = package.parent
    rows = []
    for repeat in range(repeats):
        # Reverse the second sweep to expose warmup/clock/temperature ordering effects.
        for workers in counts if repeat % 2 == 0 else list(reversed(counts)):
            print(f"\nMeasuring {workers} independent searches, repeat {repeat + 1}/{repeats}", flush=True)
            command = [str(binary), "--model", str(package), "--variant", variant,
                       "--workers", str(workers), "--seconds", str(seconds),
                       "--warmup-steps", str(warmup_steps), "--g", str(games),
                       "--b", str(batch), "--t", str(threshold), "--sims", str(sims)]
            row = measure(command, env)
            row["repeat"] = repeat + 1
            rows.append(row)
    result = {"gpu": gpu_name, "torch": torch.__version__,
              "rust": subprocess.check_output([str(cargo.parent / "rustc"), "--version"], env=env, text=True).strip(),
              "cpu_affinity_count": len(os.sched_getaffinity(0)),
              "settings": dict(g=games, b=batch, t=threshold, s=sims, variant=variant,
                               seconds=seconds, warmup_steps=warmup_steps, repeats=repeats),
              "config": settings, "runs": rows}
    (cache / "latest.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.local_entrypoint()
def main(searches: str = defaults["searches"], seconds: float = defaults["seconds"],
         repeats: int = defaults["repeats"], warmup_steps: int = defaults["warmup_steps"],
         variant: str = defaults["variant"], games: int = mcts["g"],
         batch: int = mcts["b"], threshold: int = mcts["t"], sims: int = mcts["s"],
         cpus: float = resources["cpus"], memory_gib: int = resources["memory_gib"],
         output: str = defaults["output"]):
    counts = [int(x) for x in searches.split(",")]
    if not counts or min(counts) < 1 or len(set(counts)) != len(counts):
        raise ValueError("search counts must be distinct positive integers")
    if seconds <= 0 or repeats < 1 or warmup_steps < 1 or cpus <= 0 or memory_gib < 1:
        raise ValueError("seconds, repeats, warmup-steps, cpus, and memory-gib must be positive")
    if not 1 <= games <= 65536 or batch < 1 or not 1 <= threshold <= games or sims < 1:
        raise ValueError("invalid search configuration")
    if variant not in ("puct", "exp3"):
        raise ValueError("variant must be puct or exp3")
    result = benchmark.with_options(cpu=cpus, memory=memory_gib * 1024).remote(
        counts, seconds, repeats, warmup_steps, variant, games, batch, threshold, sims,
    )
    result["resources"] = {"requested_cpus": cpus, "requested_memory_gib": memory_gib}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "runs.csv").open("w", newline="") as f:
        fields = sorted(set().union(*(r.keys() for r in result["runs"])))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["runs"])
    print("\nSearches | Model rows/s (median) | Speedup")
    baseline = statistics.median(r["model_rows_per_s"] for r in result["runs"] if r["workers"] == min(counts))
    for workers in sorted(counts):
        rate = statistics.median(r["model_rows_per_s"] for r in result["runs"] if r["workers"] == workers)
        print(f"{workers:8} | {rate:22,.0f} | {rate / baseline:.2f}x")
    print(f"Saved raw results to {output}")
