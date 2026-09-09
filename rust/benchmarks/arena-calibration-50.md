# 50-step arena calibration

The capacity formula is now `8 * G * S`, shared by `Config::default()` and the profiler through `Config::recommended_node_capacity`. Explicit caller-supplied capacities remain supported. Changing `g` or `s` via a struct update does not recompute `node_capacity`; callers constructing custom configurations should call the helper.

The original configuration peaked at 205,840 nodes (2.010 * G * S). Its initial plateau broke after game recycling; occupancy was rising again at step 50. The new capacity gives 3.98 times that observed peak. This is deliberate reserve for model variation, seed variation and longer rollouts, not an assertion of a worst-case bound. A 50-step representative-model run cannot guarantee that exhaustion is impossible for every production workload.

Both runs use PUCT, 100 simulations per move, c=1.5, epsilon=0.25, alpha=0.3 and the same grounded stub and seeds as the ten-step calibration. Steps are MCTS stepping batches, not moves per game. The current search steps all ready games once T is reached. Full-width stub evaluation is included in timings. The experimental capacity was 10 * G * S to avoid censoring the measurement; neither measurement reached it.

Reproduce from `rust` (explicit capacities reproduce the measurement index sizes):

```sh
cargo build --release --bin arena_profile
target/release/arena_profile --games 1024 --buffer 512 --t 512 --sims 100 --steps 50 --capacity 1024000
target/release/arena_profile --games 8192 --buffer 2048 --t 2048 --sims 100 --steps 50 --capacity 8192000
```

Omit `--capacity` to use the new formula. CSV files contain all 50 batches and cumulative phase timings.

## Measurements

| G / B / T | Steps | Seconds | Peak nodes | Peak / (G*S) | Final live nodes | New capacity / peak |
|---|---:|---:|---:|---:|---:|---:|
| 1024 / 512 / 512 | 50 | 24.821 | 205,840 | 2.010 | 159,613 | 3.98x |
| 8192 / 2048 / 2048 | 50 | 106.237 | 924,476 | 1.129 | 737,276 | 7.09x |

The deployment run peaked at batch 49, with 924,476 nodes (1.129 * G * S). The new budget is 7.09 times this measured peak. This trace also does not establish a long-run steady state. At its observed high-water mark, node payload plus the new index would occupy approximately 2.27 GB; the initially reserved node Vec plus index is approximately 2.54 GB, excluding the small auxiliary allocations. This is an observed-workload estimate, distinct from the filled-arena budget below.

Raw data: [original configuration](arena-steps50.csv), [deployment configuration](arena-deployment-steps50.csv).

## Deployment memory budget

For G=8192, S=100, B=T=2048:

| Allocation | Size (decimal GB) |
|---|---:|
| Logical capacity: 6,553,600 PUCT nodes, 2,168 bytes each | 14.208 |
| Hash index: 16,777,216 entries, 16 bytes each | 0.268 |
| Node Vec after geometric growth to 8,388,608 slots | 18.187 |
| Free-list allowance at 8,388,608 u32 entries | 0.034 |
| Node Vec + index + free list | 18.488 |
| Conservative transient including previous node Vec during growth | 27.582 |

Allow approximately **19 GB for a filled arena and 28 GB during growth**, plus model/runtime memory. Roots, paths, model buffers and training records add tens of MB at this configuration; allocator overhead is not measured here. These figures fit comfortably within the stated approximately 100 GB budget.

Capacity is a ceiling, not immediately resident node storage. Arena::new initially reserves at most 1,048,576 node slots and touches nodes as needed; the full hash index is initialized immediately. Freed nodes remain allocated for reuse, so memory follows the historical allocation peak, not post-sweep live count. Vec growth rounding and transient figures describe the current implementation, rather than a Rust API guarantee.

## Validation

The release profiler builds, the new formula passes the one-step smoke run with the same 7,804-node peak as before, and git diff --check passes. Existing integration tests still reference missing APIs, as documented in the initial calibration; they were not repaired as part of sizing. Search behavior and chain.rs were not changed in this follow-up.
