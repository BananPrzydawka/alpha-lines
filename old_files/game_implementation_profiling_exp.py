import importlib
import torch
import time

from config import height, width, num_parallel_games, iterations


def run_workload(GameCls):

    torch.manual_seed(0)
    g = GameCls(num_games=num_parallel_games)
    dist0 = torch.rand(num_parallel_games, height, width)
    dist1 = torch.rand(num_parallel_games, height, width)
    while not g.finished.all():
        g.distribution_step(dist0, dist1)


def profile_module(module_name):
    GameCls = importlib.import_module(module_name).batched_lines_game

    # Untimed warm-up: triggers numba JIT compilation (if applicable) and
    # general torch caching, so the profiled run reflects steady-state cost.
    run_workload(GameCls)

    print(f"\n{'=' * 80}\n{module_name}  (n={num_parallel_games})\n{'=' * 80}")


def main():
    start_time = time.perf_counter()
    for i in range(iterations):
        # profile_module("game")
        profile_module("game_experimental")
    print(f"total execution time = {time.perf_counter() - start_time}")