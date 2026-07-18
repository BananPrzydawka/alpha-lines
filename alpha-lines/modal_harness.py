import sys
import os
import inspect
import importlib
import subprocess
import shutil
from pathlib import Path
import torch
import modal
from line_profiler import LineProfiler

local_dir = Path(__file__).parent
volume = modal.Volume.from_name("alphalines", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("nsight-systems-2024.5.1", "nsight-compute-2024.1.1")
    .pip_install("torch", "numpy", "line-profiler", "torch-tb-profiler", "numba")
    .add_local_dir(local_dir, remote_path="/root")
)

app = modal.App("alphalines")

from config import gpu, timeout, host_logs, iterations


def _register_target(lp: LineProfiler, fn_ref: str):
    """
    Supported notations:
        module::function              - single function
        module::Class                 - all non-dunder methods on a class
        module::Class.method          - specific method on a class
        module::Class.method,method2  - multiple methods (comma-separated upstream)
    """
    if "::" not in fn_ref:
        raise ValueError(f"Invalid --functions entry '{fn_ref}': expected '::' separator.")

    mod_name, target = fn_ref.split("::", 1)
    mod = importlib.import_module(mod_name)

    if "." in target:
        class_name, method_name = target.split(".", 1)
        cls = getattr(mod, class_name)
        if not inspect.isclass(cls):
            raise ValueError(f"'{class_name}' in module '{mod_name}' is not a class.")
        fn = getattr(cls, method_name)
        if not callable(fn):
            raise ValueError(f"'{method_name}' on '{class_name}' is not callable.")
        lp.add_function(fn)
        print(f"  Registered method:    {mod_name}::{class_name}.{method_name}")

    else:
        obj = getattr(mod, target)
        if inspect.isclass(obj):
            registered = []
            for name, member in inspect.getmembers(obj, predicate=inspect.isfunction):
                if not name.startswith("__"):
                    lp.add_function(member)
                    registered.append(name)
            for name, member in inspect.getmembers(obj, predicate=inspect.isbuiltin):
                if not name.startswith("__"):
                    lp.add_function(member)
                    registered.append(name)
            print(f"  Registered class:     {mod_name}::{target} ({', '.join(registered)})")
        elif callable(obj):
            lp.add_function(obj)
            print(f"  Registered function:  {mod_name}::{target}")
        else:
            raise ValueError(f"'{target}' in module '{mod_name}' is not callable or a class.")


def _execute_with_profiler(
    profiler: str | None,
    keep_old_logs: bool,
    module_name: str,
    extra_functions: list[str],
    output_dir: str,
    commit_volume: bool = False,
):
    """Shared execution core — runs in both Modal remote and local contexts."""
    profiler = str(profiler).lower() if profiler else "none"

    mod = importlib.import_module(module_name)
    execute = mod.main

    if profiler == "line-profiler":
        # print("--- Executing with Line-Profiler ---")
        lp = LineProfiler()
        for fn_ref in extra_functions:
            _register_target(lp, fn_ref)
        lp_wrapper = lp(execute)
        lp_wrapper()
        lp.print_stats(stream=sys.stdout, output_unit=1e-6)

    elif profiler == "pytorch":
        # print("--- Executing with PyTorch Profiler ---")

        tb_output_dir = os.path.join(output_dir, "tb_logs")
        if not keep_old_logs and os.path.exists(tb_output_dir):
            print(f"Clearing previous PyTorch profiler logs at: {tb_output_dir}")
            shutil.rmtree(tb_output_dir)

        tmp_dir = "/tmp/tb_logs"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        # if iterations == 1:
        #     wait, warmup, active = 0, 0, 1
        # elif iterations == 2:
        #     wait, warmup, active = 0, 1, 1
        # else:
        #     wait, warmup, active = 1, 1, 1
        
        wait, warmup, active = 0, 0, 1

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tmp_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as prof:
            execute()
            prof.step()

        print(f"Profiler session finalized. Copying logs to: {tb_output_dir}")
        os.makedirs(tb_output_dir, exist_ok=True)
        for filename in os.listdir(tmp_dir):
            src = os.path.join(tmp_dir, filename)
            dst = os.path.join(tb_output_dir, filename)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

        if commit_volume:
            volume.commit()
            print("PyTorch trace saved to Modal Volume: /outputs/tb_logs")
        else:
            print(f"PyTorch trace saved locally: {tb_output_dir}")

    else:
        # print("--- Executing Standard Run (No Profiler) ---")
        execute()


def _clear_pycache(root: Path):
    removed = 0
    for d in root.rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)
        removed += 1
    print(f"--recompile: removed {removed} __pycache__ dirs under {root}")


# ── Modal path ────────────────────────────────────────────────────────────────

@app.function(image=image, volumes={"/outputs": volume}, gpu=gpu, timeout=timeout)
def run_training(profiler: str, keep_old_logs: bool, module_name: str, extra_functions: list[str] = []):
    _execute_with_profiler(profiler, keep_old_logs, module_name, extra_functions, "/outputs", commit_volume=True)


@app.local_entrypoint()
def main(
    script: str,
    profiler: str = "none",
    keep_old_logs: bool = False,
    functions: str = "",
):
    module_name = Path(script).stem
    extra_functions = [f.strip() for f in functions.split(",") if f.strip()]

    profiler_choice = profiler.lower() if profiler else "none"
    if profiler_choice == "none":
        profiler_choice = None

    valid_options = ["line-profiler", "pytorch", None]
    if profiler_choice not in valid_options:
        raise ValueError(f"Invalid profiler. Choose from: {valid_options}")

    print(f"Launching {module_name} on Modal (profiler={profiler_choice}, keep_old_logs={keep_old_logs})...")
    run_training.remote(profiler_choice, keep_old_logs, module_name, extra_functions)

    base_host_path = Path(host_logs)

    if profiler_choice == "pytorch":
        local_tb_dir = base_host_path / "tb_logs"
        if not keep_old_logs and local_tb_dir.exists():
            print(f"Clearing previous local host logs at: {local_tb_dir}")
            shutil.rmtree(local_tb_dir)
        os.makedirs(base_host_path, exist_ok=True)
        print(f"Downloading PyTorch Profiler traces to: {local_tb_dir}")
        subprocess.run(["modal", "volume", "get", "alphalines", "/tb_logs", str(base_host_path)], check=True)
        print("Download complete.")


# ── Local path (uv run modal_harness.py) ─────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaLines training harness (local mode)")
    parser.add_argument("--script", required=True, help="Target script/module to run")
    parser.add_argument("--profiler", default="none", choices=["none", "line-profiler", "pytorch"])
    parser.add_argument("--keep-old-logs", action="store_true")
    parser.add_argument("--functions", default="", help="Comma-separated line-profiler targets (module::fn)")
    parser.add_argument("--recompile", action="store_true", help="Delete all __pycache__ under the working dir before running (forces numba recompile)")
    args = parser.parse_args()

    module_name = Path(args.script).stem
    extra_functions = [f.strip() for f in args.functions.split(",") if f.strip()]
    profiler_choice = args.profiler if args.profiler != "none" else None

    base_host_path = Path(host_logs)
    os.makedirs(base_host_path, exist_ok=True)

    if args.recompile:
        _clear_pycache(local_dir)

    print(f"Running {module_name} locally (profiler={profiler_choice})...")
    _execute_with_profiler(profiler_choice, args.keep_old_logs, module_name, extra_functions, str(base_host_path))
