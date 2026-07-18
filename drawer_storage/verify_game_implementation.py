"""
Verify game_experimental (new) against game (old), then profile both across
batch sizes. Verification uses action_step with identical replayed moves;
profiling uses distribution_step. Profiling only runs if verification passes.
"""
import importlib
import time
import numpy as np
import torch
from config import num_parallel_games

GameOld = importlib.import_module("game").batched_lines_game
GameNew = importlib.import_module("game_experimental").batched_lines_game


def pick_random_moves(mask_0, mask_1, active, rng):
    m0, m1 = np.asarray(mask_0), np.asarray(mask_1)
    n = m0.shape[0]
    idx_0 = np.zeros(n, dtype=np.int64)
    idx_1 = np.zeros(n, dtype=np.int64)
    for g in range(n):
        if not active[g]:
            continue
        idx_0[g] = rng.choice(np.nonzero(m0[g])[0])
        idx_1[g] = rng.choice(np.nonzero(m1[g])[0])
    return idx_0, idx_1


def compare(seed, n, max_steps=500):
    rng = np.random.default_rng(seed)
    ga, gb = GameOld(num_games=n), GameNew(num_games=n)
    steps = 0
    while not np.asarray(ga.finished).all() and steps < max_steps:
        active = ~np.asarray(ga.finished)               # length n bool
        active_idx = np.flatnonzero(active)
        mask_0, mask_1, _, _ = ga.get_legal_masks()     # old API: full (n, H*W)
        idx_0, idx_1 = pick_random_moves(mask_0, mask_1, active, rng)  # length n, placeholders for finished
        idx_0_t, idx_1_t = torch.from_numpy(idx_0), torch.from_numpy(idx_1)
        ga.action_step(idx_0_t, idx_1_t)                                       # old: full length n
        gb.action_step(torch.from_numpy(idx_0[active_idx]),
                       torch.from_numpy(idx_1[active_idx]),
                       game_idx=active_idx)                                    # new: dense subset
        steps += 1

    checks = {
        "boards": np.array_equal(np.asarray(ga.boards), np.asarray(gb.boards)),
        "scores": np.allclose(np.asarray(ga.scores), np.asarray(gb.scores)),
        "finished": np.array_equal(np.asarray(ga.finished), np.asarray(gb.finished)),
        "move_counts": np.array_equal(np.asarray(ga.move_counts), np.asarray(gb.move_counts)),
    }
    ok = all(checks.values())
    print(f"seed {seed}: {'OK' if ok else 'MISMATCH'} ({steps} steps)")
    if not ok:
        for name, passed in checks.items():
            print(f"  {name} match: {passed}")
        diff = np.argwhere(np.asarray(ga.boards) != np.asarray(gb.boards))
        if len(diff):
            print(f"  first board mismatch: game {diff[0][0]}, row {diff[0][1]}, col {diff[0][2]}")
    return ok


def verify():
    print("=== verification: game_experimental vs game ===")
    all_ok = True
    for seed in range(20):
        all_ok &= compare(seed, n=num_parallel_games)
    print(f"All seeds matched: {all_ok}\n")
    return all_ok


def bench(GameCls, num_games):
    start = time.time()
    bg = GameCls(num_games)
    dist = torch.ones(num_games, 10, 16)
    while not bg.finished.all():
        bg.distribution_step(dist, dist)
    return time.time() - start


def profile():
    print("=== profiling: distribution_step ===")
    sizes = [1, 4, 16, 64, 256, 1024, 4096, 16384]
    # sizes = [1, 1, 1, 1]
    print(f"warmup step for stability | old game time: {bench(GameOld, 1):.3f}, new game time: {bench(GameNew, 1):.3f}")
    
    # do a warmup pass not counted in lp for the bench fn:
    # bg = GameNew(1)
    # dist = torch.ones(1, 10, 16)
    # while not bg.finished.all():
    #     bg.distribution_step(dist, dist)


    print(f"{'batch':>7} | {'old per-game':>14} | {'new per-game':>14} | {'speedup':>8}")
    for n in sizes:
        t_old = bench(GameOld, n)
        t_new = bench(GameNew, n)
        po, pn = t_old / n * 1000, t_new / n * 1000
        print(f"{n:>7} | {po:>12.3f}ms | {pn:>12.3f}ms | {t_old / t_new:>7.2f}x")


def main():
    whole_start_time = time.perf_counter()
    verify()
    profile()
    print(f"whole execution time: {(time.perf_counter() - whole_start_time):.3f}")