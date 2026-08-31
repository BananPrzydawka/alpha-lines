# alpha-lines — Rust implementations

Two implementations of the game in `main/game.py`:

- **`game.rs` + `game_kernels.rs`** — a straight, 1:1 port. No algorithmic changes, no
  optimizations. This is the baseline that everything else is measured against.
- **`incremental.rs`** — level-based incremental scoring. Same results, ~8x faster at
  scoring.

## Layout

```
rust/
  src/
    config.rs          board geometry from main/config.py
    rng.rs             seedable PRNG standing in for numba's np.random
    game_kernels.rs    port of main/game_kernels.py        <- reference, untouched
    game.rs            port of main/game.py                <- reference, untouched
    incremental.rs     level-based incremental scorer      <- the fast implementation
    bin/verify.rs      replays a move file and dumps state, for the cross-check
    bin/bench.rs       performance driver
    bin/profile.rs     batch-size sweep, per stage
    bin/order.rs       isolates batch size from memory access order
    bin/phases.rs      per-function breakdown of a rollout
  tests/
    kernels.rs         25 unit tests on the kernel layer
    game.rs            22 tests on the game API + a randomized invariant rollout
    incremental.rs     21 tests on the incremental scorer + differential rollouts
  xcheck/
    xcheck.py          runs an implementation against main/game.py and diffs the output
    bench.py           runs both against main/game.py and tabulates
```

The crate has **zero dependencies** on purpose: a reference implementation should build and
run identically anywhere without a crates.io fetch.

## Build and test

```sh
cd rust
cargo build --release
cargo test --release
```

## Cross-check against the Python implementation

```sh
uv run python rust/xcheck/xcheck.py --games 300 --seed 0
uv run python rust/xcheck/xcheck.py --games 300 --seed 0 --impl incremental
```

Both implementations replay an identical, externally generated sequence of random legal
moves over a few hundred games, and dump every observable piece of state after every move.
The dumps are compared **exactly** — no float tolerance, since every float involved is
either 0/1, an integer-valued score, or the same single IEEE division on both sides.

Compared per game, per move:

| what | detail |
| --- | --- |
| `boards` | full (N, H, W) board state |
| `scores` | both players |
| `move_counts`, `finished` | |
| `mask_p0`, `mask_p1` | full legal-move masks |
| `count_p0`, `count_p1` | legal-move counts |
| `get_encoded_states(0/1)` | both players' full 7-channel encodings |
| `get_terminal_outcomes()` | win/draw/loss codes, at the end |
| `print_state()` | rendered text, both players, every step |
| `import_prints()` | state reconstructed from that text, both players |

The dump directory is a fresh `/tmp/alpha-lines-xcheck-*`, deleted on success and **kept on
failure** so a mismatch can be inspected. Pass `--out-dir DIR` to choose it, or `--keep` to
hold on to it either way. The dumps run ~40 MB per 300-game run, so clear stale ones out
after a debugging session.

Result, for **both** implementations: **identical on every field, at every step**, over
300–512 games × 6–8 seeds (~2800 games each, ~43 moves per game).

## Benchmark against the Python implementation

```sh
uv run python rust/xcheck/bench.py --games 2048 --reps 60
```

Both sides run the same measurements with the same batch size, the same fixed random policy
distributions, and the same mid-game board state. numba's JIT compilation is warmed up and
excluded. Both are single-threaded, because `main/config.py` sets
`game_kernels_parralel = False`.

Measured at 2048 games, 60 reps (median of repeated runs; spread under 5%):

| benchmark | python (numba) | rust reference | rust incremental | ref/py | incr/py |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rollout_total` (2048 games to completion) | 488 ms | 458 ms | **177 ms** | 1.1x | 2.8x |
| `rollout_per_step` | 11.3 ms | 10.4 ms | **4.01 ms** | 1.1x | 2.8x |
| `apply_and_score` (scoring alone) | 8.37 ms | 7.32 ms | **906 us** | 1.1x | 9.2x |
| `legal_masks_kernel` | 619 us | 1.28 ms | – | 0.5x | – |
| `score_batch_both_players` | 6.77 ms | 5.82 ms | – | 1.2x | – |
| `sample_move_kernel` | 1.21 ms | 1.19 ms | – | 1.0x | – |
| `get_encoded_states` | 4.92 ms | 2.26 ms | – | 2.2x | – |

**The straight port is not meaningfully faster than the Python** — 1.0–1.1x on a full
rollout. numba is an LLVM compiler, the kernels are already tight scalar loops over small
boards, and there is no interpreter overhead left to win back.

**The incremental scorer is where the win is: 8.1x over the reference on scoring alone**,
2.8x on a whole rollout. `apply_and_score` isolates the part that actually differs, by
replaying a pre-recorded move sequence so masks and sampling are factored out.

Notes on the individual rows:

- `score_batch` dominates the reference. `apply_and_score_kernel` rescores both players from
  scratch after every move, so `score_player` runs `2 x active_games` times per step, and it
  allocates four `H*W` buffers on each call. It is ~60% of a rollout step in both languages.
  That is exactly what `incremental.rs` replaces.
- `legal_masks_kernel` is the one row where numba wins outright, by 2x. numba auto-vectorizes
  the mask stores (its generated assembly uses AVX: 58 `%ymm` references, 65 `vmovups`); the
  straight Rust translation emits scalar stores. Confirmed by elimination — it is not bounds
  checks, not allocation, and not first-touch page faults (slicing the buffers per game,
  swapping the `f64` counters for integers, and hoisting the allocation out of the timed
  region all changed nothing).
- `get_encoded_states` is a clear Rust win, 2.1x, because the Python builds the 7-channel
  tensor with five full-array boolean-mask assignments plus a channel swap that copies,
  where the Rust makes a single pass.

## The incremental scorer (`src/incremental.rs`)

The reference rescores a board from scratch after every move: a flood fill from the border
plus two full diagonal scans, twice per active game. `incremental.rs` maintains the same
score incrementally, so a move usually costs a handful of neighbour lookups instead.

### What is stored

Per cell, alongside the board value, a **level**: the number of steps to the nearest border
cell walking only over that player's own marks. Border marks are level 0, their neighbours
1, and so on; a mark with no route to the border is `INF`.

Only `(r + c)` even squares are playable and orthogonal neighbours always have the opposite
parity, so the reference's 8-connected flood fill is exactly 4-connectivity over the
diagonals.

### Why levels and not "reachable" bits

A single reachable bit cannot survive deletion. A line anchored to the border at *both*
ends, cut in the middle, leaves every surviving cell reachable — but a bit gives no way to
know that without re-running the flood fill.

Storing *which neighbour I depend on* does not work either. Without back-edges the
dependency graph is a BFS tree, which does not record alternate routes, so deletion wrongly
kills cells that still have a path. With back-edges it records them but gains cycles, and a
cycle is a set of cells that mutually justify each other with nothing underneath: in a
diamond A–B–C–D where only A touches the border, deleting A leaves C and D pointing at each
other, both still claiming to be reachable.

Levels fix this because the provider relation is **derived, not stored**: `n` is a valid
provider for `x` exactly when `level[n] < level[x]`. Strict inequality makes cycles
impossible — every mark is held up by something strictly closer to the border, and the chain
has to bottom out there.

### The invariant everything rests on

`level[x] != INF` if and only if `x` has a path to the border. It is uniform across a
connected component and therefore uniform along any diagonal run. **The score moves only
when a cell crosses between INF and finite.** Level changes that stay finite cost work and
change no score — that is the tax for never running a global flood fill.

### The four paths

|  | condition | cost |
| --- | --- | --- |
| insert, fast | the new mark touches no dead blob | O(1) |
| insert, slow | it revives a dead blob | O(component) |
| remove, fast | no neighbour sat exactly one level above | O(1) |
| remove, slow | something may have been cut | O(component) |

Both fast paths use the same local delta: a run of length L scores L if L >= 2, so filling a
gap between runs of length `la` and `lb` gains `1 + (la == 1) + (lb == 1)`, with both sides
clamped to `{0, 1, >=2}` — one point for extending a run at all, plus one more for each side
that was a lone mark. Removal is the same, negated.

The slow paths measure the affected components before and after and apply the difference.
`relax_down` pushes improved levels outward after an insertion (levels only fall);
`repair_up` re-derives levels after a removal (levels only rise), processing in increasing
level order so a cell is only re-examined once everything that could still justify it has
settled.

### The pathological case

A severed cycle has no external ground, so its cells climb in lockstep — 3,3,4 then 5,5,6 —
one level per round, until they exceed the longest possible real path and become INF. It
self-terminates without any cycle detection, but it is the worst case for this scheme.

The **level cap is implemented**: `MAX_LEVEL` is the playable-square count (80 here), and any
candidate above it becomes INF immediately, which bounds the climb at 80 rounds. The
alternative mitigation — bail out to a local flood fill over the component once a repair
touches more than some threshold of cells — is **not** implemented.

**This is the dominant remaining cost.** Under random self-play at 2048 games, `repair_up`
pops 2.14M cells against `relax_down`'s 137k, a 15.7:1 ratio, and that work is concentrated
in the 15.7% of removals that take the slow path — about **567 pops per slow removal**,
which is the lockstep climb and nothing else. Component walks, by contrast, cost only 1.2
cells per operation. So the bailout is the obvious next optimization.

Reproduce with `cargo build --release --features stats && ./target/release/profile`.

### How it is verified

- **21 tests** in `tests/incremental.rs`, including the diamond severed cycle, a line
  anchored at both ends cut in the middle, a line anchored at one end cut in the middle, a
  dead blob revived, and collisions cutting both players at once.
- **Differential rollouts** against the reference: identical boards, scores, move counts and
  finished flags after every move, over 12 seeds x 128 games, plus a variant that forces
  collisions in half the games so the removal paths are exercised heavily.
- **`check_invariants`** asserts, after every move in the slower tests, that every stored
  level matches both the local rule and a from-scratch BFS, and that both running scores
  match the reference scorer.
- **The Python cross-check**, via `--impl incremental`.

## Batch-size profile (`src/bin/profile.rs`, `order.rs`, `phases.rs`)

### What is batched and what is not

| stage | shape |
| --- | --- |
| `legal_masks_kernel` | batched: one call per step over all games, allocates 2 x N x H x W f32 |
| `sample_move_kernel` | batched: one call per step per player |
| `get_encoded_states` | batched: one call, allocates N x 7 x H x W f32 |
| reference `apply_and_score_kernel` | loop over games; per game a full O(H*W) rescore |
| incremental `apply_step` | loop over games; per game a few scattered cell lookups |

Storage is batched throughout — boards, levels and scores are flat arrays indexed by game.
The incremental scorer itself is strictly **per game, scalar**: `insert` / `remove` operate
on one game's 160-cell slice. Its one genuine batching benefit is that a single `Scratch` is
reused across every game and move, so the stamp array, the DFS stack and the bucket queue
stay hot in L1 instead of being allocated per game.

### Does batch size change per-game cost?

Mostly no. `./target/release/order` settles it: the same batch and the same recorded moves,
replayed in two traversal orders that must produce identical state (asserted).

```
step-major:  for each step { for each game { apply } }    <- what the game loop does
game-major:  for each game { for each step { apply } }    <- one game start to finish
```

Incremental scorer, ns per active game-move:

| games | step-major | game-major | ratio |
| ---: | ---: | ---: | ---: |
| 16 | 628 ns | 651 ns | 0.96x |
| 64 | 553 ns | 527 ns | 1.05x |
| 256 | 548 ns | 562 ns | 0.98x |
| 1024 | 570 ns | 576 ns | 0.99x |
| 2048 | 572 ns | 593 ns | 0.96x |
| 8192 | 551 ns | 603 ns | 0.91x |

**Flat in both batch size and traversal order.** The incremental scorer touches ~10 scattered
cells per move, so it is latency-bound on a handful of loads whether the board is hot or
cold.

Reference scorer, same experiment:

| games | step-major | game-major | ratio |
| ---: | ---: | ---: | ---: |
| 1 | 2411 ns | 2289 ns | 1.05x |
| 64 | 3755 ns | 2867 ns | 1.31x |
| 1024 | 3886 ns | 3044 ns | 1.28x |
| 8192 | 3934 ns | 2960 ns | 1.33x |

**The reference is genuinely cache-sensitive, by about 1.3x.** It streams a whole 160-byte
board per rescore, so visiting N boards round-robin evicts each one before it comes back;
game-major order recovers the loss at every batch size.

#### A measurement trap worth knowing about

Do not read anything from a batch of one. Timing the 2048 games individually gives:

```
per-game cost, incremental (ns per move):
  min 213   p10 342   median 466   mean 547   p90 859   max 2512
  3% of games come in under 300 ns/move on their own
```

A 12x spread, because a game's cost depends entirely on how many slow removals it happens to
hit. A single game tells you nothing about the batch.

An earlier version of this section reported n=1 at 257 ns/move against ~520 ns at larger
batches and called it a batch-size penalty. That figure was one game — the same game, at the
same seed, measured repeatedly, which makes it look reproducible without making it
representative. It happened to fall in the cheap 3% tail. Sampling n=1 the honest way, over
512 *independent* single-game batches:

```
n=1 averaged over 512 independent single-game batches: 643 ns per move
```

Which is **slower** than the batched figure, not twice as fast. So the direction of the
effect is the opposite of what was first reported: batching the incremental scorer is worth
roughly 15% per game, and the gain comes from the shared hot `Scratch`.

(The n=1 row in the reference table above is a single game too, and should not be read as a
trend either. The step-major vs game-major comparison within each row is still valid, since
both sides replay the identical games.)

### Per-function breakdown

`distribution_step` is four separable pieces of work. `./target/release/phases` times each
one across batch sizes. Microseconds per completed game; `sum` is the four timed phases and
`untimed` is the same rollout with no timers in it, so the two agreeing means the
decomposition is not lying.

**rust reference**

| games | active | masks | sample | apply | sum | untimed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9 | 12.9 | 38.4 | **119.0** | 171.2 | 166.0 |
| 4 | 0.2 | 11.7 | 35.7 | **129.8** | 177.5 | 178.3 |
| 16 | 0.1 | 12.1 | 39.5 | **151.0** | 202.8 | 202.6 |
| 64 | 0.0 | 12.7 | 43.8 | **157.8** | 214.4 | 211.5 |
| 256 | 0.0 | 12.7 | 44.3 | **157.9** | 214.9 | 213.3 |
| 1024 | 0.0 | 22.9 | 46.3 | **160.0** | 229.2 | 240.9 |
| 2048 | 0.0 | 24.8 | 49.9 | **166.5** | 241.2 | 248.2 |
| 8192 | 0.0 | 18.0 | 50.5 | **166.3** | 234.8 | 233.3 |

**rust incremental**

| games | active | masks | sample | apply | sum | untimed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9 | 14.2 | **39.5** | 28.0 | 82.6 | 81.1 |
| 4 | 0.2 | 12.3 | **38.0** | 25.5 | 76.1 | 83.7 |
| 16 | 0.1 | 13.6 | **49.5** | 29.1 | 92.3 | 84.3 |
| 64 | 0.0 | 12.8 | **43.6** | 22.6 | 79.0 | 82.0 |
| 256 | 0.0 | 13.3 | **46.7** | 22.0 | 82.0 | 81.7 |
| 1024 | 0.0 | 13.5 | **47.1** | 21.5 | 82.1 | 81.3 |
| 2048 | 0.0 | 14.2 | **50.0** | 21.3 | 85.5 | 80.6 |
| 8192 | 0.0 | 18.5 | **48.6** | 21.5 | 88.5 | 91.1 |

**Why the reference slows down.** Almost all of it is `apply`: **119 -> 166 us, +47 us**,
against +12 us for sampling and +5 for masks. That is `score_player` rescoring a whole board
from scratch, twice per game per move, with the eviction problem described above.

**Why the incremental one does not.** Its two batch effects run in opposite directions and
roughly cancel: `apply` gets **cheaper** (28.0 -> 21.3 us, -24%) from the shared hot
`Scratch`, while `sample` gets **more expensive** (39.5 -> 50.0 us) because the two
distributions are 640 bytes each at n=1 and 1.3 MB each at n=2048. Net 82.6 -> 85.5 us.

**Where to optimize next.** For the incremental implementation the scorer is no longer the
problem. At n=2048 the split is **sample 50 us (58%), apply 21 us (25%), masks 14 us (17%)**.
`sample_move_kernel` is now the single largest cost in a rollout, and it is untouched
reference code that makes two full passes over all 160 cells per game per player.

### End-to-end throughput

Microseconds per completed game — the opening move to a finished board, including masks,
sampling and scoring (`uv run python rust/xcheck/bench.py --sweep`). Small batch sizes are
averaged over enough independent rollouts to cover ~2048 games.

| games | rollouts | python (numba) | rust reference | rust incremental | incr vs py |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2048 | 646.6 us | 169.5 us | 77.5 us | 8.34x |
| 2 | 1024 | 424.6 us | 175.8 us | 75.5 us | 5.62x |
| 4 | 512 | 339.2 us | 182.2 us | 72.5 us | 4.68x |
| 16 | 128 | 287.2 us | 219.2 us | 76.9 us | 3.73x |
| 64 | 32 | 271.5 us | 232.9 us | 81.3 us | 3.34x |
| 256 | 8 | 254.5 us | 238.2 us | 82.0 us | 3.10x |
| 1024 | 2 | 250.9 us | 229.4 us | 82.4 us | 3.04x |
| 2048 | 1 | 246.8 us | 238.7 us | 87.9 us | 2.81x |
| 4096 | 1 | 265.2 us | 237.3 us | 84.5 us | 3.14x |
| 8192 | 1 | 253.8 us | 227.9 us | 83.6 us | 3.04x |

The three implementations respond to batch size in three different directions:

- **Python needs the batch: 2.6x from n=1 to n=2048.** All of it is amortizing per-call
  dispatch over bigger arrays — four numba calls per step cost the same whether they cover
  one game or two thousand. This is the reason the game is written batched at all.
- **The reference gets *worse* with the batch, ~1.4x**, from the eviction effect.
- **The incremental scorer is flat**, 77 to 88 us across four orders of magnitude.

Which is why the incremental version's lead over Python shrinks from 8.3x to 3x as the batch
grows: batching helps Python and does nothing for Rust.

### Would a SIMD or GPU formulation help?

Split by control flow, not by data:

- `legal_masks_kernel`, `get_encoded_states` — uniform per-cell control flow, pure
  elementwise work. **Vectorize well.**
- `sample_move_kernel` — a reduction then a fixed-trip-count scan. **Vectorizes** with some
  effort.
- reference `score_player` — the flood fill is divergent, but the two diagonal scans are
  fixed-trip-count. **Partially vectorizable**, and uniform across games.
- incremental `apply_move` — **maximally divergent.** Each move takes one of five paths, and
  their costs differ by three orders of magnitude (a dead insert is ~4 lookups; a slow
  removal is ~567 repair pops). Lanes would run at worst-case cost.

That is the real trade this algorithm makes: **it swaps regularity for work.** It does ~7.4x
less work, but irregularly. On a CPU that is an unambiguous win. On a GPU it may not be —
slow removals are ~2% of operations, so in a 32-lane warp there is a ~47% chance at least one
lane hits one and stalls the other 31. The reference's uniform per-game cost is far friendlier
to that execution model, which is worth weighing before porting either one to CUDA.

## Faithfulness notes

Behaviour that looks like a bug but is reproduced deliberately, because the Python does it:

- **The collision blast overwrites stones.** When both players pick the same square, that
  square and all eight neighbours of playable parity are set to `REMOVED_SQUARE`
  unconditionally — including squares already holding a player's mark.
- **The sampler's degenerate branch ignores the mask.** If the masked distribution sums to
  under `1e-8`, `sample_move_kernel` falls back to a uniform draw over the *whole* board,
  legal or not.
- **The sampler keeps scanning after it has chosen.** The accumulation loop runs to the end
  of the board rather than breaking out.
- **`print_state` blanks out squares the first-move rule forbids.** On move 0, channel 1 is
  zeroed over the half the player may not open in, so `np.argmax` over the first five
  channels returns 0 and those playable squares render as `"  "`. `import_prints` puts them
  back via the parity rule.
- **`import_prints` cannot recover the move count.** It only distinguishes "untouched" (0)
  from "played" (1).
- **`get_encoded_states` overwrites channel 1.** The one-hot pass writes it, then the
  playable mask overwrites it.

The one intentional structural difference is the **RNG**. numba's kernels draw from a hidden
per-thread `np.random` state that cannot be reproduced bit-for-bit outside numba, so `rng.rs`
supplies an explicit xoshiro256++ instead. The port keeps the *structure* of the draws
identical — same count, same order, same call sites — and everything the cross-check compares
is driven by explicit actions, never by these draws. `BatchedLinesGame::new` and
`import_prints` therefore take a seed argument that the Python constructor does not have.

## Data formats used by the cross-check

Little-endian throughout.

`moves.bin` — written by `xcheck.py`, read by `verify`:

```
"ALMV", u32 version, u32 n_games, u32 height, u32 width, u32 n_steps
n_steps * n_games * 2 * i32          # (step, game, player) -> flat action index
```

`state_*.bin` / `reimport_*.bin` — written by both sides, diffed by `xcheck.py`:

```
"ALST", u32 version, u32 n_games, u32 height, u32 width, u32 n_records
per record:
  n*H*W    i8    boards
  n*2      f32   scores
  n        i32   move_counts
  n        u8    finished
  n*H*W    f32   mask_p0
  n*H*W    f32   mask_p1
  n        f32   count_p0
  n        f32   count_p1
  n*7*H*W  f32   encoding, player 0
  n*7*H*W  f32   encoding, player 1
trailer: u8 has_outcomes, then if set: n i64 outcomes_p0, n i64 outcomes_p1
```
