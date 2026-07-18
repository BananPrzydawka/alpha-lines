import time
from game_experimental import batched_lines_game as Game
from game_kernels import legal_masks_kernel
from config import height, width, num_parallel_games

def main():
    g = Game(num_games=num_parallel_games)
    legal_masks_kernel(g.boards, g.move_counts, g.half_width, height, width)  # warm up

    t0 = time.perf_counter()
    for _ in range(100):
        legal_masks_kernel(g.boards, g.move_counts, g.half_width, height, width)
    print((time.perf_counter() - t0) / 100 * 1000, "ms/call")

    print(legal_masks_kernel.nopython_signatures)
    print(legal_masks_kernel.signatures)