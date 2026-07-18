import time
from pathlib import Path
import modal

# 1. Define the CPU container environment (Adding numpy silences the PyTorch C++ warning)
local_dir = Path(__file__).parent
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_dir(local_dir, remote_path="/root")
)

app = modal.App("lines-game-cpu-benchmark")

NUM_GAMES_SERIAL = 2000
NUM_GAMES_BATCHED = 2000

# --- Core Benchmark Functions ---

def run_serial_workload(num_games):
    """Pure Python native game instances loop sequentially."""
    from game_slop import lines_game
    import torch
    
    policy_dummy = torch.ones(10, 16)
    start_time = time.perf_counter()
    
    for _ in range(num_games):
        game = lines_game()
        while not game.finished:
            game.make_move_from_distributions(policy_dummy, policy_dummy)
            
    return time.perf_counter() - start_time


def run_batched_workload(num_games, device_str):
    """Vectorized PyTorch batch tracking processing games simultaneously."""
    from game_slop import batched_lines_game
    import torch
    
    # Warmup instance
    warmup_bg = batched_lines_game(10, device=device_str)
    dist_warmup = torch.ones(10, 10, 16, device=device_str)
    warmup_bg.step(dist_warmup, dist_warmup)
    
    start_time = time.perf_counter()
    bg = batched_lines_game(num_games, device=device_str)

    while not bg.finished.all():
        dist = torch.ones(num_games, 10, 16, device=device_str)
        bg.step(dist, dist)

    return time.perf_counter() - start_time


# 2. Remote execution driver running on Modal CPU instance
@app.function(image=image, cpu=4.0)  
def run_remote_benchmark(local_results: dict):
    # Profile remote setup/imports
    start_import = time.perf_counter()
    import torch
    remote_import_time = time.perf_counter() - start_import

    # Execute workloads
    remote_serial_time = run_serial_workload(NUM_GAMES_SERIAL)
    remote_batched_time = run_batched_workload(NUM_GAMES_BATCHED, "cpu")

    # Generate Output Report
    print("\n" + "=" * 65)
    print(" APPLES-TO-APPLES CPU PERFORMANCE COMPARISON")
    print("=" * 65)
    print(f"{'Workload Profile':<35} | {'Local Host':<15} | {'Modal CPU':<15}")
    print("-" * 65)
    print(f"{'Library Import (s)':<35} | {local_results['import_time']:<15.4f} | {remote_import_time:<15.4f}")
    
    # Fixed alignment using exact global variable casing and nested f-strings
    print(f"{f'Serial Pure-Py (s) [{NUM_GAMES_SERIAL}]':<35} | {local_results['serial_time']:<15.4f} | {remote_serial_time:<15.4f}")
    print(f"{f'Batched PyTorch (s) [{NUM_GAMES_BATCHED}]':<35} | {local_results['batched_time']:<15.4f} | {remote_batched_time:<15.4f}")
    print("-" * 65)
    
    serial_diff = (remote_serial_time - local_results['serial_time']) / local_results['serial_time']
    batched_diff = (remote_batched_time - local_results['batched_time']) / local_results['batched_time']
    
    print(f"Serial Clock Performance Delta:  {serial_diff * 100:+.2f}%")
    print(f"Vectorized Memory Performance Delta: {batched_diff * 100:+.2f}%")
    print("=" * 65 + "\n")


# 3. Local orchestrator entrypoint
@app.local_entrypoint()
def main():
    print("Initializing local CPU profiles...")
    
    start_import = time.perf_counter()
    import torch
    local_import_time = time.perf_counter() - start_import

    # Force game logic metrics onto local CPU context
    local_serial_time = run_serial_workload(NUM_GAMES_SERIAL)
    local_batched_time = run_batched_workload(NUM_GAMES_BATCHED, "cpu")

    local_results = {
        "import_time": local_import_time,
        "serial_time": local_serial_time,
        "batched_time": local_batched_time
    }

    print("Local CPU metrics collected. Transferring container logic to Modal...")
    run_remote_benchmark.remote(local_results)