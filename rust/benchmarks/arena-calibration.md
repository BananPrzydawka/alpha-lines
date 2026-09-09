# Initial arena calibration

Run from `rust`:

```sh
cargo run --release --bin arena_profile -- --games 1024 --buffer 512 --t 512 --sims 100 --steps 10 --capacity 204800
```

PUCT defaults: c=1.5, epsilon=0.25, alpha=0.3. Grounded model copied from chain: gradient decay 0.9, lognormal prior noise sigma 0.75, coverage-based tanh values with Gaussian noise sigma 0.3. Seeds: search 0xA1FA, model 0xE1A4. One seed is an initial measurement, not a statistical bound.

A step is one call to Search::step once at least T games are ready. The current implementation advances every ready game, not exactly T. The profiler defers automatic stepping solely to measure occupancy before and after the sweep. Ten batches completed 6,095 game moves (mean 5.952, range 1–10). Slot-order collection favors earlier games.

The one-step pilot took 0.425 s, suggesting approximately 4–10 s for ten steps. The corrected ten-step run took 5.474 s: model 2.861 s, collection/scatter/backup 2.471 s, step/sweep 0.138 s. Model timing includes all B rows even when underfilled, matching production. These are wall-clock phase timings, not hardware sampling or real neural inference timings.

| Step | Games advanced | Before sweep | After sweep |
|---:|---:|---:|---:|
| 1 | 1024 | 7,804 | 4,525 |
| 2 | 513 | 50,081 | 38,814 |
| 3 | 728 | 89,681 | 55,726 |
| 4 | 647 | 107,205 | 73,827 |
| 5 | 591 | 125,455 | 86,293 |
| 6 | 522 | 137,944 | 93,823 |
| 7 | 519 | 145,483 | 102,333 |
| 8 | 519 | 153,987 | 107,082 |
| 9 | 514 | 158,736 | 111,617 |
| 10 | 518 | 163,270 | 113,567 |

Peak live nodes and allocated slots both reached 163,270. At 2,168 bytes per PUCT node, touched node storage alone is about 337.6 MiB, excluding index, free list, roots, buffers and allocator overhead. The arena reserves capacity for 204,800 nodes; freeing nodes recycles slots without shrinking storage.

No steady state is established: pre-sweep occupancy increased 4,534 nodes in the last batch, and post-sweep occupancy increased 1,950. The G*S/2*1.25 formula gives 64,000 nodes, 2.55 times below the observed peak. Keep 204,800 as experimental headroom for this ten-step case; do not treat it as a validated long-rollout capacity. No production default was changed.

The earlier run with the incorrect per-game stopping condition was excluded from these results. It continued beyond ten batches and eventually hit the 204,800 capacity, further ruling out treating this short-run capacity as a long-run bound.

Validation: release binary builds and completes the requested run without reaching capacity. Full cargo test --release fails to compile existing tests referring to removed APIs (including IdWrite, diag, last_depth). rustfmt is not installed in the local toolchain. Existing edits to chain.rs and search.rs were preserved.
