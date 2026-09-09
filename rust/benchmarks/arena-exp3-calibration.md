# EXP3 arena calibration, 50 MCTS steps

Run from `rust`:

```sh
cargo build --release --bin arena_profile
target/release/arena_profile --exp3 --games 1024 --buffer 512 --t 512 --sims 100 --steps 50
```

EXP3 gamma=0.1, grounded model, search seed 0xA1FA, model seed 0xE1A4. Capacity defaults to 819,200 nodes (`8 * G * S`). Same model and stepping semantics as the PUCT measurements. EXP3 initializes log weights from the model priors and uses its exploration floor rather than PUCT root Dirichlet noise. Because the variants consume random numbers differently, the seeds do not imply matching trajectories or identical model draws for corresponding positions.

| Metric | EXP3 | Earlier PUCT |
|---|---:|---:|
| MCTS steps | 50 | 50 |
| Runtime, seconds | 22.553 | 24.821 |
| Peak live nodes / allocated slots | 97,863 | 205,840 |
| Final live nodes after sweep | 11,066 | 159,613 |
| Bytes per node | 1,520 | 2,168 |
| Peak node payload, decimal MB | 148.752 | 446.261 |
| Mean moves per game | 26.249 | 28.313 |
| Min–max moves per game | 1–49 | 4–49 |

EXP3 used 52.5% fewer peak nodes in this run. Its maximum occurred at stepping batch 43. Final pre-sweep occupancy was 62,497 nodes; the last sweep dropped 51,431. The default arena capacity was 8.37 times the peak; no exhaustion occurred. Phase timings: model 14.221 s, search 8.070 s, step/sweep 0.248 s.

These are counts for 50 MCTS stepping batches, not equal game progress. Fixed slot-order collection advances some games much more than others, particularly with EXP3: some games stayed at zero moves through early batches. This is existing search behavior, not a profiler scheduling change. Peak memory follows allocated slots and is retained for reuse after sweeping. The node-payload numbers exclude index, roots, buffers, allocator rounding and other runtime storage.

This single representative-model seed is not a worst-case bound. Keep the conservative shared capacity formula unchanged. Raw data: [arena-exp3-steps50.csv](arena-exp3-steps50.csv).

Validation: release binary builds; all 50 stepping batches complete below capacity; profiler diff has no whitespace errors. The pre-existing integration-test compilation issues remain outside this change.
