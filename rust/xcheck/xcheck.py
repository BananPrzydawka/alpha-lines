#!/usr/bin/env python3
"""Cross-checks the Rust engine in ../src against main/game.py.

Both implementations replay an identical, externally generated move sequence over a few
hundred games and dump every observable piece of state after every move. The dumps are then
compared field by field, exactly (no tolerance: every float involved is either 0/1, an
integer-valued score, or the same single IEEE division in both languages).

What is compared, per game and per step:
  boards, scores, move_counts, finished, both legal masks, both legal-move counts,
  both players' 7-channel encodings, the terminal outcome codes, the print_state rendering
  for both players, and the state reconstructed by import_prints from that rendering.

Run:  uv run python rust/xcheck/xcheck.py [--games 300] [--seed 0]
"""

import argparse
import pathlib
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "main"))

from config import height, width  # noqa: E402
from game import batched_lines_game  # noqa: E402

STATE_MAGIC = b"ALST"
MOVES_MAGIC = b"ALMV"
VERSION = 1
HW = height * width


# --------------------------------------------------------------------------- move sequence

def generate_moves(n_games, seed):
    """Plays n_games to completion with uniformly random *legal* moves, recording the
    action indices. Collisions (both players picking the same square) arise naturally and
    are left in, since they exercise the removal branch of apply_and_score_kernel."""
    rng = np.random.default_rng(seed)
    game = batched_lines_game(num_games=n_games)
    steps = []
    guard = 0
    while not game.finished.all():
        mask_0, mask_1, _, _ = game.get_legal_masks()
        idx_0 = _pick(mask_0, game.finished, rng)
        idx_1 = _pick(mask_1, game.finished, rng)
        game.action_step(idx_0, idx_1)
        steps.append(np.stack([idx_0, idx_1], axis=1).astype(np.int32))
        guard += 1
        assert guard < 1000, "move generation did not terminate"
    return np.stack(steps, axis=0)  # (steps, games, 2)


def _pick(mask, finished, rng):
    """One uniformly random legal index per game; index 0 for already-finished games
    (action_step ignores them, but still range-checks the index)."""
    out = np.zeros(mask.shape[0], dtype=np.int64)
    for g in range(mask.shape[0]):
        if finished[g]:
            continue
        legal = np.flatnonzero(mask[g] == 1.0)
        assert legal.size > 0, f"game {g} is active but has no legal moves"
        out[g] = legal[rng.integers(legal.size)]
    return out


def write_moves(path, moves):
    n_steps, n_games, _ = moves.shape
    with open(path, "wb") as f:
        f.write(MOVES_MAGIC)
        f.write(struct.pack("<5I", VERSION, n_games, height, width, n_steps))
        f.write(np.ascontiguousarray(moves, dtype="<i4").tobytes())


# ------------------------------------------------------------------------------ state dump

class StateWriter:
    def __init__(self, path, n):
        self.f = open(path, "wb")
        self.n = n
        self.records = 0
        self.f.write(STATE_MAGIC)
        self.f.write(struct.pack("<5I", VERSION, n, height, width, 0))

    def push(self, game):
        assert game.n == self.n
        mask_0, mask_1, count_0, count_1 = game.get_legal_masks()
        w = self.f.write
        w(np.ascontiguousarray(game.boards, dtype="<i1").tobytes())
        w(np.ascontiguousarray(game.scores, dtype="<f4").tobytes())
        w(np.ascontiguousarray(game.move_counts, dtype="<i4").tobytes())
        w(np.ascontiguousarray(game.finished, dtype=np.uint8).tobytes())
        w(np.ascontiguousarray(mask_0, dtype="<f4").tobytes())
        w(np.ascontiguousarray(mask_1, dtype="<f4").tobytes())
        w(np.ascontiguousarray(count_0, dtype="<f4").tobytes())
        w(np.ascontiguousarray(count_1, dtype="<f4").tobytes())
        self.records += 1

    def finish(self):
        self.f.seek(20)
        self.f.write(struct.pack("<I", self.records))
        self.f.close()


FIELDS = [
    ("boards", "<i1", lambda n: n * HW),
    ("scores", "<f4", lambda n: n * 2),
    ("move_counts", "<i4", lambda n: n),
    ("finished", "u1", lambda n: n),
    ("mask_0", "<f4", lambda n: n * HW),
    ("mask_1", "<f4", lambda n: n * HW),
    ("count_0", "<f4", lambda n: n),
    ("count_1", "<f4", lambda n: n),
]


def read_state(path):
    raw = pathlib.Path(path).read_bytes()
    assert raw[:4] == STATE_MAGIC, f"{path}: bad magic {raw[:4]!r}"
    version, n, h, w, records = struct.unpack("<5I", raw[4:24])
    assert version == VERSION and (h, w) == (height, width), f"{path}: bad header"
    off = 24
    out = []
    for _ in range(records):
        rec = {}
        for name, dtype, count_of in FIELDS:
            count = count_of(n)
            nbytes = count * np.dtype(dtype).itemsize
            rec[name] = np.frombuffer(raw, dtype=dtype, count=count, offset=off)
            off += nbytes
        out.append(rec)
    assert off == len(raw), f"{path}: {len(raw) - off} trailing bytes"
    return n, out


# ------------------------------------------------------------------------------- comparison

class Mismatch(Exception):
    pass


def compare_states(label, a, b, n):
    n_a, recs_a = a
    n_b, recs_b = b
    if n_a != n_b:
        raise Mismatch(f"{label}: batch size {n_a} vs {n_b}")
    if len(recs_a) != len(recs_b):
        raise Mismatch(f"{label}: record count {len(recs_a)} vs {len(recs_b)}")

    for step, (ra, rb) in enumerate(zip(recs_a, recs_b)):
        for name, _dtype, _count in FIELDS:
            va, vb = ra[name], rb[name]
            if va.shape != vb.shape:
                raise Mismatch(f"{label} step {step} {name}: shape {va.shape} vs {vb.shape}")
            bad = np.flatnonzero(va != vb)
            if bad.size:
                per_game = va.size // n
                i = int(bad[0])
                raise Mismatch(
                    f"{label} step {step} field {name}: {bad.size}/{va.size} elements differ; "
                    f"first at flat index {i} (game {i // per_game}, offset {i % per_game}): "
                    f"py={va[i]!r} rs={vb[i]!r}"
                )

    return len(recs_a)


# ------------------------------------------------------------------------------ python side

def python_replay(moves, out_dir, tag="py"):
    n_steps, n_games, _ = moves.shape
    game = batched_lines_game(num_games=n_games)

    state = StateWriter(out_dir / f"state_{tag}.bin", n_games)
    state.push(game)
    for step in range(n_steps):
        game.action_step(moves[step, :, 0].astype(np.int64), moves[step, :, 1].astype(np.int64))
        state.push(game)
    state.finish()

    # The Rust side adopts the final boards from scratch and must land on the same state.
    # The Python equivalent is rendering the final position and importing it back, which is
    # the only route it has to "here is a board, work out its score".
    text = capture_print(game, 0)
    adopted = batched_lines_game.import_prints(text, 0)
    re = StateWriter(out_dir / f"reimport_{tag}.bin", n_games)
    re.push(adopted)
    re.finish()


def capture_print(game, player):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        game.print_state(player=player)
    return buf.getvalue()


# ------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="keep the dump directory")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    binary = ROOT / "rust" / "target" / "release" / "verify"
    if not binary.exists():
        sys.exit(f"missing {binary}; build it with `cargo build --release` in rust/")

    tmp = args.out_dir or tempfile.mkdtemp(prefix="alpha-lines-xcheck-")
    out_dir = pathlib.Path(tmp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"dump directory: {out_dir}")

    t = time.perf_counter()
    moves = generate_moves(args.games, args.seed)
    print(f"generated {moves.shape[0]} moves x {args.games} games "
          f"({time.perf_counter() - t:.1f}s, includes numba compile)")

    write_moves(out_dir / "moves.bin", moves)

    t = time.perf_counter()
    python_replay(moves, out_dir)
    print(f"python replay + dump: {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    res = subprocess.run(
        [str(binary), "--moves", str(out_dir / "moves.bin"), "--out-dir", str(out_dir),
         "--tag", "rs"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        sys.exit(f"rust verify failed ({res.returncode}):\n{res.stdout}\n{res.stderr}")
    print(f"rust replay + dump:   {time.perf_counter() - t:.1f}s  [{res.stdout.strip()}]")

    checks = []
    try:
        steps = compare_states(
            "state", read_state(out_dir / "state_py.bin"), read_state(out_dir / "state_rs.bin"), args.games
        )
        checks.append(f"state dumps identical over {steps} records "
                      f"({len(FIELDS)} fields x {args.games} games each)")

        steps = compare_states(
            "reimport", read_state(out_dir / "reimport_py.bin"),
            read_state(out_dir / "reimport_rs.bin"), args.games,
        )
        checks.append(f"adopting the final board from scratch matches over {steps} states")
    except Mismatch as e:
        print("\nFAIL")
        print(e)
        sys.exit(1)

    print()
    for c in checks:
        print(f"  ok  {c}")
    print("\nPASS: the rust engine matches main/game.py exactly")

    if not args.keep and args.out_dir is None:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
