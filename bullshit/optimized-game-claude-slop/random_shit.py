import torch
import time
from game import lines_game, BatchedLinesGame

# --- Configuration ---
NUM_GAMES = 100

def test_random_serial(num_games):
    """Plays games sequentially using Python-only game."""
    start_time = time.time()
    policy = torch.ones(10, 16)

    for _ in range(num_games):
        game = lines_game()
        while not game.finished:
            game.make_move_from_distributions(policy, policy)

    return time.time() - start_time

def test_random_batched(num_games):
    """Plays games in parallel using BatchedLinesGame."""
    start_time = time.time()
    bg = BatchedLinesGame(num_games)
    dist = torch.ones(num_games, 10, 16)

    while not bg.finished.all():
        bg.step(dist, dist)

    return time.time() - start_time

# --- Execution and Results ---
print(f"Testing {NUM_GAMES} games\n" + "-"*40)

t_rand_serial = test_random_serial(NUM_GAMES)
t_rand_batch = test_random_batched(NUM_GAMES)

print(f"Random Serial:  {t_rand_serial:.4f}s")
print(f"Random Batched: {t_rand_batch:.4f}s")
print(f"Speedup:        {t_rand_serial/t_rand_batch:.2f}x")
