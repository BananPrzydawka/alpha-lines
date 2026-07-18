import time
from pathlib import Path
import modal

local_dir = Path(__file__).parent
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_dir(local_dir, remote_path="/root")
)

app = modal.App("lines-game-hardware-sweep")

# L4 max > 25k games, < 55k games
# 
NUM_GAMES_BATCHED = 50000

def run_batched_workload(num_games, device_str):
    """Vectorized PyTorch engine running on the targeted hardware layer."""
    from game_slop import batched_lines_game
    import torch
    
    device = torch.device(device_str)
    
    # Warmup context initialization to completely isolate memory allocation out of the loop
    warmup_bg = batched_lines_game(10, device=device.type)
    dist_warmup = torch.ones(10, 10, 16, device=device)
    warmup_bg.step(dist_warmup, dist_warmup)
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    bg = batched_lines_game(num_games, device=device.type)

    while not bg.finished.all():
        dist = torch.ones(num_games, 10, 16, device=device)
        bg.step(dist, dist)

    if device.type == "cuda":
        torch.cuda.synchronize()
        
    return time.perf_counter() - start_time


# --- Modal Cloud Targets ---

@app.function(image=image, gpu="A100-40GB")
def run_remote_gpu(num_games: int):
    """Runs the tracking workload on an Nvidia GPU."""
    return run_batched_workload(num_games, "cuda")


# --- Local Control Entrypoint ---

@app.local_entrypoint()
def main():
    import torch
    print(f"Starting LOCAL evaluations ({NUM_GAMES_BATCHED} parallel games)...")
    
    # Local GPU Execution (Checks if CUDA is present on your laptop)
    local_gpu_available = torch.cuda.is_available()
    if local_gpu_available:
        print(f"Evaluating Local Host GPU ({torch.cuda.get_device_name(0)})...")
        local_gpu_time = run_batched_workload(NUM_GAMES_BATCHED, "cuda")
    else:
        print("Local Host GPU not available or CUDA not initialized. Skipping layer.")
        local_gpu_time = None

    print("\nDispatching workloads to Modal Cloud Clusters...")
    
    # Trigger remote cloud execution routes asynchronously
    cloud_gpu_future = run_remote_gpu.spawn(NUM_GAMES_BATCHED)
    
    print("Awaiting Cloud GPU (Nvidia L4) termination...")
    remote_gpu_time = cloud_gpu_future.get()
    
    # --- Throughput and Metrics Calculations ---
    r_gpu_tp = NUM_GAMES_BATCHED / remote_gpu_time
    
    l_gpu_tp = NUM_GAMES_BATCHED / local_gpu_time if local_gpu_time else 0.0
    l_gpu_str = f"{local_gpu_time:<15.4f}s" if local_gpu_time else f"{'N/A':<15}"
    l_gpu_tp_str = f"{l_gpu_tp:.2f} games/s" if local_gpu_time else "N/A"

    # --- Consolidated Full Hardware Breakdown Report ---
    print("\n" + "=" * 60)
    print(f" NUMBER OF GAMES: {NUM_GAMES_BATCHED}")
    print(f"{'Hardware Target Profile':<27} | {'Execution Time':<15} | {'Throughput'}")
    print(f"{'Local Host GPU':<27} | {l_gpu_str} | {l_gpu_tp_str}")
    print(f"{'Modal Cloud GPU (Nvidia L4)':<27} | {remote_gpu_time:<15.4f}s | {r_gpu_tp:.2f} games/s")
    print("=" * 60 + "\n")