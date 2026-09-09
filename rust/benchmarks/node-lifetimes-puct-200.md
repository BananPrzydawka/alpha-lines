# PUCT node lifetimes over 200 MCTS steps

Configuration: G=1024, B=T=512, S=100, PUCT c=1.5, epsilon=0.25, alpha=0.3. Same grounded model and seeds as the previous arena measurements. Default capacity: 819,200 nodes. A step is a stepping batch, advancing all ready games when T is reached.

From `rust`:

```sh
cargo build --release --bin arena_profile
target/release/arena_profile --games 1024 --buffer 512 --t 512 --sims 100 --steps 200 --lifetimes benchmarks/node-lifetimes-puct-200.csv > benchmarks/arena-puct-steps200.csv
python3 benchmarks/summarize_lifetimes.py benchmarks/node-lifetimes-puct-200.csv
```

Files:

- `node-lifetimes-puct-200.csv`: exact integer-age buckets at every step, separately for live and removed nodes. Zero-count buckets are included.
- `node-lifetimes-puct-200.buckets.csv`: ranges 0, 1, 2–3, 4–7, 8–15, 16–31, 32–63, 64–127, 128–255, 256+.
- `node-lifetimes-puct-200.summary.csv`: counts, mean, nearest-rank p50/p90/p99, maximum age, and tail counts >=16/32/64/128 for both populations at every step.
- `arena-puct-steps200.csv`: occupancy, stepping counts and cumulative phase timings.

## Counter semantics

Each node has a u32 `sweep_age`, initially zero and reset on slot reuse. Each sweep increments it only if the node survives, saturating at u32::MAX. A node removed at its first sweep has lifetime zero; one removed at its second has lifetime one. Live ages are measured after the sweep. Private roots are not arena nodes and are excluded.

The profiler snapshots the age histogram before stepping, then measures survivors afterward. Removed count at age a equals pre-sweep count at a minus post-sweep survivors at a+1. Search::step performs no arena insertion, making this subtraction exact. Assertions reconcile removed counts at every step. Counters advance on every global MCTS sweep even when a node's owning games do not advance. The u32 fits existing struct padding: PUCT nodes remain 2,168 bytes on this build.

Live ages are unfinished lifetimes (right-censored). The removed distribution describes completed lifetimes at that particular step, not cumulative deaths. The live distribution is biased toward long-lived nodes; do not interpret its percentiles as percentiles of all completed node lifetimes. Counts concern survival, not how recently a node was accessed or whether it was useful to the last search.

## Results

Completed in 103.764 seconds, with peak allocation 227,519 nodes and final live count 116,392. No capacity exhaustion. Timings include measurement overhead; model, search and step phase timers exclude histogram scans and serialization outside those phases.

At step 200:

| Population | Count | Mean age | p50 | p90 | p99 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Live after sweep | 116,392 | 22.682 | 23 | 34 | 143 | 199 |
| Removed this sweep | 54,167 | 1.200 | 0 | 3 | 16 | 72 |

The live tail includes 17,468 nodes aged >=32, 8,603 aged >=64, and 1,186 aged >=128. Across the entire run, the longest completed lifetime was 185 survived sweeps, removed at step 187. Nodes still live at age 199 might survive longer beyond this measurement window.

Per-slot cumulative moves at the end range from 11 to 196, mean 111.536. These counts include moves across replacement games; they are not current game ply. Slot-order collection and retained claims from slowly advancing slots can contribute to the long tail. This measurement does not establish that current game ply has become uniformly distributed; checking that hypothesis requires recording current ply per slot separately.

## Validation

Two focused release tests pass: recycled slots reset age, and age is unchanged during collection/evaluation but increments for sweep survivors. All 400 histogram populations reconcile with the corresponding live/deleted arena counts across 200 steps. The first 50 steps exactly match prior PUCT occupancy and stepping counts, confirming instrumentation did not alter that trace. Existing unrelated integration-test compilation failures remain outside this change.
