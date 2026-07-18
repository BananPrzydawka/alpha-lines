import sys
import os
import subprocess
import shutil
from pathlib import Path
import torch
import modal
from line_profiler import LineProfiler

# 1. Setup local directory and developer CUDA container image with versioned profiling tools
local_dir = Path(__file__).parent
volume = modal.Volume.from_name("alphalines", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("nsight-systems-2024.5.1", "nsight-compute-2024.1.1")
    .pip_install("torch", "numpy", "line-profiler", "torch-tb-profiler")
    .add_local_dir(local_dir, remote_path="/root")
)

app = modal.App("alphalines")

# 2. Local imports
from config import gpu, timeout, host_logs
from test_utils import execute_training_loop


def get_profiler_path(binary_name: str) -> str:
    """ Dynamically walks the installation directory to find versioned profiler binaries """
    search_root = "/opt/nvidia"
    if os.path.exists(search_root):
        for root, dirs, files in os.walk(search_root):
            if binary_name in files:
                candidate = os.path.join(root, binary_name)
                if os.access(candidate, os.X_OK):
                    return candidate
                    
    cuda_bin = f"/usr/local/cuda/bin/{binary_name}"
    if os.path.exists(cuda_bin):
        return cuda_bin
        
    return binary_name


# 3. Remote function configured with Volume mount for system artifact collection
@app.function(image=image, volumes={"/outputs": volume}, gpu=gpu, timeout=timeout)
def run_training(profiler: str, keep_old_logs: bool):
    profiler = str(profiler).lower() if profiler else "none"

    if profiler == "line-profiler":
        print("--- Executing with Line-Profiler ---")
        lp = LineProfiler()
        lp_wrapper = lp(execute_training_loop)
        lp_wrapper()
        lp.print_stats(stream=sys.stdout)

    elif profiler == "pytorch":
        print("--- Executing with PyTorch Profiler ---")
        
        if not keep_old_logs:
            if os.path.exists("/outputs/tb_logs"):
                print("Clearing previous PyTorch profiler logs from volume...")
                shutil.rmtree("/outputs/tb_logs")
                
        local_tmp_dir = "/tmp/tb_logs"
        if os.path.exists(local_tmp_dir):
            shutil.rmtree(local_tmp_dir)
        os.makedirs(local_tmp_dir, exist_ok=True)
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(local_tmp_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False
        ) as prof:
            execute_training_loop(prof=prof)
            
        print("Profiler session finalized. Synchronizing logs to Modal Volume...")
        os.makedirs("/outputs/tb_logs", exist_ok=True)
        for filename in os.listdir(local_tmp_dir):
            source_file = os.path.join(local_tmp_dir, filename)
            destination_file = os.path.join("/outputs/tb_logs", filename)
            if os.path.isdir(source_file):
                shutil.copytree(source_file, destination_file, dirs_exist_ok=True)
            else:
                shutil.copy2(source_file, destination_file)
        
        volume.commit()
        print("PyTorch trace saved to Modal Volume: /outputs/tb_logs")

    elif profiler in ("nsys", "ncu"):
        print(f"--- Executing CLI Binary: {profiler} ---")
        
        reports_dir = "/outputs/reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        if not keep_old_logs:
            print(f"Clearing previous {profiler} logs from volume...")
            prefix = "nsys_trace" if profiler == "nsys" else "ncu_analysis"
            for item in os.listdir(reports_dir):
                if item.startswith(prefix):
                    full_path = os.path.join(reports_dir, item)
                    if os.path.isdir(full_path):
                        shutil.rmtree(full_path)
                    else:
                        os.remove(full_path)
                        
        inline_code = "from test_utils import execute_training_loop; execute_training_loop()"
        
        if profiler == "nsys":
            # Direct override to use the specific architecture binary layout
            target_binary = "/opt/nvidia/nsight-systems/2024.5.1/target-linux-x64/nsys"
            if not os.path.exists(target_binary):
                target_binary = get_profiler_path("nsys")
                
            cmd = [
                target_binary, "profile",
                "-o", "/outputs/reports/nsys_trace",
                "-t", "cuda,nvtx",                  # Explicitly intercept CUDA driver and NVTX event markers
                "--sample=none",                    # Disables the privilege-restricted CPU sampling engine
                "--trace-fork-before-exec=true",    # Tracks across process fork barriers
                "--force-overwrite=true",
                "--stats=true",
                "python", "-c", inline_code
            ]
        else:
            binary_path = get_profiler_path(profiler)
            cmd = [
                binary_path,
                "-o", "/outputs/reports/ncu_analysis",
                "--target-processes", "all",
                "--force-overwrite",
                "python", "-c", inline_code
            ]
            
        print(f"Executing command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd="/root")
        volume.commit()
        print(f"NVIDIA report saved to Modal Volume: /outputs/reports")

    else:
        print("--- Executing Standard Run (No Profiler) ---")
        execute_training_loop()


@app.local_entrypoint()
def main(profiler: str = "none", keep_old_logs: bool = False):
    profiler_choice = profiler.lower() if profiler else "none"
    if profiler_choice == "none":
        profiler_choice = None

    valid_options = ["line-profiler", "pytorch", "nsys", "ncu", None]
    if profiler_choice not in valid_options:
        raise ValueError(f"Invalid profiler. Choose from: {valid_options}")
        
    print(f"Launching training job on Modal (Selected Profiler Mode: {profiler_choice} | Keep Logs: {keep_old_logs})...")
    
    # 1. Run remote cloud function (Container tracking and billing terminates here)
    run_training.remote(profiler_choice, keep_old_logs)
    
    base_host_path = Path(host_logs)
    
    # 2. Automatically handle local host cleanup and structured file downloads
    if profiler_choice == "pytorch":
        local_tb_dir = base_host_path / "tb_logs"
        if not keep_old_logs and local_tb_dir.exists():
            print(f"Clearing previous local host logs at: {local_tb_dir}")
            shutil.rmtree(local_tb_dir)
            
        os.makedirs(base_host_path, exist_ok=True)
        print(f"Downloading PyTorch Profiler traces automatically to: {local_tb_dir}")
        subprocess.run(["modal", "volume", "get", "alphalines", "/tb_logs", str(base_host_path)], check=True)
        print("Download complete.")

    elif profiler_choice in ("nsys", "ncu"):
        local_reports_dir = base_host_path / "reports"
        if not keep_old_logs and local_reports_dir.exists():
            print(f"Clearing previous local host reports at: {local_reports_dir}")
            shutil.rmtree(local_reports_dir)
            
        os.makedirs(base_host_path, exist_ok=True)
        print(f"Downloading NVIDIA profiling reports automatically to: {local_reports_dir}")
        subprocess.run(["modal", "volume", "get", "alphalines", "/reports", str(base_host_path)], check=True)
        print("Download complete.")