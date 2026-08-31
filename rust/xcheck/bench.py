#!/usr/bin/env python3
"""Benchmarks main/game.py against the Rust implementations on identical workloads.

Both sides run the same measurements, with the same batch size, the same fixed random
policy distributions, and the same mid-game board state (reached by playing `--warmup-moves`
random moves from the start):

  rollout_total             N games played to completion via distribution_step
  rollout_per_step          the same, divided by the number of moves it took
  apply_and_score           scoring alone: a pre-recorded move sequence replayed through
                            the apply path, so masks and sampling are factored out
  legal_masks_kernel        one call over the whole batch
  score_batch_both_players  one call per player over the whole batch
  sample_move_kernel        one call over the whole batch
  get_encoded_states        one call over the whole batch, player 0

numba compilation is excluded: every kernel is called once to warm the JIT before timing.

Run:  uv run python rust/xcheck/bench.py [--games 2048] [--reps 20]
      uv run python rust/xcheck/bench.py --sweep
"""

import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main"))

from config import height, width  # noqa: E402
from game import batched_lines_game  # noqa: E402
from game_kernels import (  # noqa: E402
    PLAYER_0_MARK, PLAYER_1_MARK, apply_and_score_kernel, legal_masks_kernel,
    sample_move_kernel, score_batch,
)

# (row label, python key, rust reference key, rust incremental key or None)
ORDER = [
    ("rollout_total", "rollout_total", "rollout_total", "rollout_incremental_total"),
    ("rollout_per_step", "rollout_per_step", "rollout_per_step", "rollout_incremental_per_step"),
    ("apply_and_score", "apply_and_score", "apply_and_score_reference",
     "apply_and_score_incremental"),
    ("legal_masks_kernel", "legal_masks_kernel", "legal_masks_kernel", None),
    ("score_batch_both_players", "score_batch_both_players", "score_batch_both_players", None),
    ("sample_move_kernel", "sample_move_kernel", "sample_move_kernel", None),
    ("get_encoded_states", "get_encoded_states", "get_encoded_states", None),
]


def timeit(fn, reps):
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return time.perf_counter() - t


def run_python(n, reps, warmup_moves, seed):
    rng = np.random.default_rng(seed)
    dist_p0 = rng.random((n, height, width), dtype=np.float32)
    dist_p1 = rng.random((n, height, width), dtype=np.float32)

    # warm the JIT on a throwaway batch so compilation is not timed
    warm = batched_lines_game(num_games=2)
    warm.distribution_step(dist_p0[:2], dist_p1[:2])
    warm.get_encoded_states(0)
    warm.get_legal_masks()
    score_batch(warm.boards, PLAYER_0_MARK, height, width)
    sample_move_kernel(dist_p0[:2], np.ones((2, height, width), np.float32),
                       np.ones(2, np.bool_), height, width)

    results = {}

    game = batched_lines_game(num_games=n)
    t = time.perf_counter()
    steps = 0
    while not game.finished.all():
        game.distribution_step(dist_p0, dist_p1)
        steps += 1
        assert steps < 1000, "rollout did not terminate"
    rollout = time.perf_counter() - t
    results["rollout_total"] = (rollout, 1)
    results["rollout_per_step"] = (rollout, steps)

    mid = batched_lines_game(num_games=n)
    for _ in range(warmup_moves):
        mid.distribution_step(dist_p0, dist_p1)
    active = ~mid.finished
    mask_0, _, _, _ = legal_masks_kernel(mid.boards, mid.move_counts, mid.half_width, height, width)

    # scoring alone, with masks and sampling factored out: record a move sequence, then
    # time replaying it through apply_and_score_kernel
    recorded = []
    rec_game = batched_lines_game(num_games=n)
    while not rec_game.finished.all():
        # note: not `active`, which the sampler benchmark below closes over
        rec_active = ~rec_game.finished
        m0, m1 = rec_game._raw_masks()
        r0, c0 = sample_move_kernel(dist_p0, m0, rec_active, height, width)
        r1, c1 = sample_move_kernel(dist_p1, m1, rec_active, height, width)
        apply_and_score_kernel(rec_game.boards, rec_game.move_counts, rec_game.finished,
                               rec_game.scores, r0, c0, r1, c1, rec_active, height, width)
        recorded.append((r0, c0, r1, c1, rec_active))

    def replay():
        g = batched_lines_game(num_games=n)
        for r0, c0, r1, c1, act in recorded:
            apply_and_score_kernel(g.boards, g.move_counts, g.finished, g.scores,
                                   r0, c0, r1, c1, act, height, width)

    results["apply_and_score"] = (timeit(replay, 1), len(recorded))

    results["legal_masks_kernel"] = (
        timeit(lambda: legal_masks_kernel(mid.boards, mid.move_counts, mid.half_width,
                                          height, width), reps), reps)
    results["score_batch_both_players"] = (
        timeit(lambda: (score_batch(mid.boards, PLAYER_0_MARK, height, width),
                        score_batch(mid.boards, PLAYER_1_MARK, height, width)), reps), reps)
    results["sample_move_kernel"] = (
        timeit(lambda: sample_move_kernel(dist_p0, mask_0, active, height, width), reps), reps)
    results["get_encoded_states"] = (
        timeit(lambda: mid.get_encoded_states(0), reps), reps)

    return results, steps


def run_rust(n, reps, warmup_moves, seed):
    binary = ROOT / "rust" / "target" / "release" / "bench"
    if not binary.exists():
        sys.exit(f"missing {binary}; build it with `cargo build --release` in rust/")
    res = subprocess.run(
        [str(binary), "--games", str(n), "--reps", str(reps),
         "--warmup-moves", str(warmup_moves), "--seed", str(seed)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        sys.exit(f"rust bench failed:\n{res.stdout}\n{res.stderr}")
    results = {}
    steps = None
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "#":
            steps = int(parts[4])
            continue
        results[parts[0]] = (float(parts[1]), int(parts[2]))
    return results, steps


def fmt(seconds):
    if seconds >= 1.0:
        return f"{seconds:8.3f} s "
    if seconds >= 1e-3:
        return f"{seconds * 1e3:8.3f} ms"
    return f"{seconds * 1e6:8.3f} us"


# ------------------------------------------------------------------- batch-size sweep

def rollout_python(n, seed, reps):
    """`reps` independent rollouts of n games each. Returns (seconds, games played)."""
    rng = np.random.default_rng(seed)
    dist_p0 = rng.random((n, height, width), dtype=np.float32)
    dist_p1 = rng.random((n, height, width), dtype=np.float32)
    games = [batched_lines_game(num_games=n) for _ in range(reps)]  # outside the timer
    t = time.perf_counter()
    for game in games:
        while not game.finished.all():
            game.distribution_step(dist_p0, dist_p1)
    return time.perf_counter() - t, n * reps


def rollout_rust(n, seed, reps):
    binary = ROOT / "rust" / "target" / "release" / "bench"
    res = subprocess.run(
        [str(binary), "--games", str(n), "--reps", "1", "--seed", str(seed),
         "--rollout-reps", str(reps)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        sys.exit(f"rust bench failed:\n{res.stdout}\n{res.stderr}")
    out, games = {}, None
    for line in res.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "#":
            games = int(parts[6])
        else:
            out[parts[0]] = float(parts[1])
    return out["rollout_total"], out["rollout_incremental_total"], games


def sweep(sizes, seed, target_games=2048):
    """Microseconds per completed game, for all three implementations.

    At small batch sizes one rollout is a sample of a single game, whose cost varies by
    more than 10x, so every size is averaged over enough independent rollouts to cover
    ~`target_games` games.
    """
    print("microseconds per completed game (opening move to a finished board)\n")
    head = (f"{'games':>7} {'rollouts':>9} {'sampled':>8}  {'python':>10} "
            f"{'rust ref':>10} {'rust incr':>10}   {'incr vs py':>10}")
    print(head)
    print("-" * len(head))
    for n in sizes:
        reps = max(1, target_games // n)
        py, py_games = rollout_python(n, seed, reps)
        rs_ref, rs_inc, rs_games = rollout_rust(n, seed, reps)
        assert py_games == rs_games, f"{py_games} vs {rs_games}"
        per = lambda t: t / py_games * 1e6
        print(f"{n:>7} {reps:>9} {py_games:>8}  {per(py):9.1f}us {per(rs_ref):9.1f}us "
              f"{per(rs_inc):9.1f}us   {py / rs_inc:9.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup-moves", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--sweep", action="store_true",
                    help="rollout throughput across batch sizes, all three implementations")
    args = ap.parse_args()

    if args.sweep:
        # warm the JIT so numba compilation is not charged to the first size
        warm = batched_lines_game(num_games=2)
        d = np.ones((2, height, width), dtype=np.float32)
        warm.distribution_step(d, d)
        sweep([1, 2, 4, 16, 64, 256, 1024, 2048, 4096, 8192], args.seed)
        return

    print(f"games={args.games}  reps={args.reps}  board={height}x{width}")
    print("both sides single-threaded (main/config.py sets game_kernels_parralel = False)\n")

    py, py_steps = run_python(args.games, args.reps, args.warmup_moves, args.seed)
    rs, rs_steps = run_rust(args.games, args.reps, args.warmup_moves, args.seed)
    print(f"rollout length: python {py_steps} moves, rust {rs_steps} moves\n")

    head = (f"{'benchmark':<26} {'python':>12} {'rust ref':>12} {'rust incr':>12} "
            f"{'ref/py':>8} {'incr/py':>9}")
    print(head)
    print("-" * len(head))
    for label, pk, rk, ik in ORDER:
        p = py[pk][0] / py[pk][1]
        r = rs[rk][0] / rs[rk][1]
        if ik is None:
            print(f"{label:<26} {fmt(p)} {fmt(r)} {'-':>12} {p / r:7.1f}x {'-':>9}")
        else:
            i = rs[ik][0] / rs[ik][1]
            print(f"{label:<26} {fmt(p)} {fmt(r)} {fmt(i)} {p / r:7.1f}x {p / i:8.1f}x")

    ref_apply = rs["apply_and_score_reference"]
    inc_apply = rs["apply_and_score_incremental"]
    speedup = (ref_apply[0] / ref_apply[1]) / (inc_apply[0] / inc_apply[1])
    print(f"\nincremental vs reference scoring, in Rust: {speedup:.1f}x")


if __name__ == "__main__":
    main()
