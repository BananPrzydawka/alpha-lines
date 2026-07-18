import time
import numpy as np
import torch

from config import device, num_parallel_games, mcts_num_simulations, iterations
from game import batched_lines_game
from model import alpha_lines_net
from mcts_exp3 import BatchedExp3MCTS


def _make_move(game, policies):
    idx_0 = np.array([p[0].argmax() for p in policies])
    idx_1 = np.array([p[1].argmax() for p in policies])
    game.action_step(idx_0, idx_1)


@torch.no_grad()
def main():
    model = alpha_lines_net().to(device).eval()
    mcts = BatchedExp3MCTS(num_sims=mcts_num_simulations)

    # warmup: numba kernel compile + cuda/model init, excluded from timing
    warm = batched_lines_game(num_games=num_parallel_games)
    _make_move(warm, mcts.search(warm, model))

    game = batched_lines_game(num_games=num_parallel_games)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        _make_move(game, mcts.search(game, model))
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    nodes = iterations * mcts_num_simulations * num_parallel_games
    print(f"games={num_parallel_games} sims={mcts_num_simulations} iters={iterations} device={device}")
    print(f"{nodes} nodes in {elapsed:.3f}s  ->  {nodes / elapsed:,.0f} nodes/s")