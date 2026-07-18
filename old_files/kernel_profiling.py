# /// script
# requires-python = ">=3.11"
# dependencies = ["numba", "numpy"]
# ///
"""
Pinpoints why game_experimental's apply_and_score_kernel is slower than game's.

Two effects are confounded in the full-game benchmark:
  (1) codegen: gather-indexing boards[active_idx[i]] vs direct boards[g]. If
      _score_player is inlined, indirect indexing can defeat LLVM aliasing/
      vectorization for the WHOLE inlined body -> constant-factor slowdown even
      when the accessed rows are contiguous.
  (2) memory locality: a scattered active_idx gathers non-contiguous board rows.

This script separates them:
  COND A  all games active, active_idx = arange(n)  (contiguous, identical work)
          -> any gap here is (1), pure codegen. This is the decisive test.
  COND B  half active, scattered active_idx          (adds gather pattern)
          -> extra gap over A is (2), locality.

line_profiler is NOT used (it inflates Numba boundary times). Warm kernels +
perf_counter, board template restored outside the timed region.
"""
import os, time, numpy as np, numba

# match your run: single value so both modules compile the same way
THREADS = int(os.environ.get("BENCH_THREADS", numba.config.NUMBA_NUM_THREADS))
numba.set_num_threads(THREADS)

from config import height, width
import game_kernels as K            # direct-index (active bool mask)
import game_experimental_kernels as E  # gather-index (active_idx)

PLAYABLE = K.PLAYABLE_SQUARE
rng = np.random.default_rng(0)


def make_template(n):
    """Mid-game boards with marks so _score_player has real flood-fill work."""
    r = np.arange(height)[:, None]
    c = np.arange(width)[None, :]
    playable = (r + c) % 2 == 0
    boards = np.zeros((n, height, width), dtype=np.int8)
    boards[:, playable] = PLAYABLE
    # sprinkle player marks on ~40% of playable cells, deterministically per game
    pr, pc = np.where(playable)
    for g in range(n):
        sel = rng.random(pr.size) < 0.4
        marks = rng.integers(K.PLAYER_0_MARK, K.PLAYER_1_MARK + 1, size=sel.sum())
        boards[g, pr[sel], pc[sel]] = marks.astype(np.int8)
    return boards


def make_moves(n):
    r0 = rng.integers(0, height, n).astype(np.int64)
    c0 = rng.integers(0, width, n).astype(np.int64)
    r1 = rng.integers(0, height, n).astype(np.int64)
    c1 = rng.integers(0, width, n).astype(np.int64)
    return r0, c0, r1, c1


def time_kernel(call, boards, template, mc, sc, fin, reps):
    total = 0.0
    for _ in range(reps):
        np.copyto(boards, template)          # restore OUTSIDE timed region
        mc.fill(0); sc.fill(0); fin.fill(False)
        t0 = time.perf_counter()
        call()
        total += time.perf_counter() - t0
    return total / reps


def run(n, reps=30):
    template = make_template(n)
    r0, c0, r1, c1 = make_moves(n)          # full-length (indexed by g)

    boards = template.copy()
    mc = np.zeros(n, np.int32); sc = np.zeros((n, 2), np.float32); fin = np.zeros(n, np.bool_)

    # ---- COND A: all active, contiguous ----
    active_bool = np.ones(n, np.bool_)
    idx_all = np.arange(n, dtype=np.int64)
    d = lambda: K.apply_and_score_kernel(boards, mc, fin, sc, r0, c0, r1, c1, active_bool, height, width)
    g = lambda: E.apply_and_score_kernel(boards, mc, fin, sc, r0, c0, r1, c1, idx_all, height, width)
    d(); g()  # warm (both signatures)
    tA_d = time_kernel(d, boards, template, mc, sc, fin, reps)
    tA_g = time_kernel(g, boards, template, mc, sc, fin, reps)

    # ---- COND B: half active, scattered ----
    idx_sub = np.sort(rng.choice(n, n // 2, replace=False)).astype(np.int64)
    active_sub = np.zeros(n, np.bool_); active_sub[idx_sub] = True
    r0s, c0s, r1s, c1s = r0[idx_sub], c0[idx_sub], r1[idx_sub], c1[idx_sub]  # dense for E
    dB = lambda: K.apply_and_score_kernel(boards, mc, fin, sc, r0, c0, r1, c1, active_sub, height, width)
    gB = lambda: E.apply_and_score_kernel(boards, mc, fin, sc, r0s, c0s, r1s, c1s, idx_sub, height, width)
    dB(); gB()
    tB_d = time_kernel(dB, boards, template, mc, sc, fin, reps)
    tB_g = time_kernel(gB, boards, template, mc, sc, fin, reps)

    print(f"n={n:6d}  reps={reps}")
    print(f"  A all-active contiguous  direct={tA_d*1e3:8.3f}ms  gather={tA_g*1e3:8.3f}ms  ratio={tA_g/tA_d:5.2f}x  <- codegen only")
    print(f"  B half-active scattered  direct={tB_d*1e3:8.3f}ms  gather={tB_g*1e3:8.3f}ms  ratio={tB_g/tB_d:5.2f}x  <- +locality")
    # per-active-game, so A and B are comparable
    print(f"     per-active-game        direct={tB_d/(n//2)*1e6:6.2f}us gather={tB_g/(n//2)*1e6:6.2f}us")
    print()


def vector_report():
    """Count SIMD instructions in each compiled kernel. Fewer in the gather
    version confirms indirect indexing suppressed vectorization."""
    def simd(fn):
        try:
            asm = "\n".join(fn.inspect_asm().values())
        except Exception as e:
            return f"n/a ({e})"
        v = sum(asm.count(m) for m in ("vmovup", "vmulp", "vaddp", "vfmadd", "vpadd", "vpcmpeq"))
        return f"{v} SIMD-ish mnemonics"
    print("compiled-code vectorization:")
    print("  game_kernels.apply_and_score_kernel          :", simd(K.apply_and_score_kernel))
    print("  game_experimental.apply_and_score_kernel     :", simd(E.apply_and_score_kernel))
    print("  game_kernels._score_player                   :", simd(K._score_player))
    print("  game_experimental._score_player              :", simd(E._score_player))
    print()


def main():
    print(f"threads={numba.get_num_threads()}  board={height}x{width}\n")
    for n in (256, 4096, 16384):
        run(n)
    vector_report()
    print("Reading: if COND A ratio >> 1, the cause is codegen (gather indexing"
          "\ndefeats vectorization of inlined _score_player), NOT thread balancing"
          "\nor wasted work. COND B minus A ratio is the extra memory-locality cost.")