# Search reset and numerical hardening

Call `search.flush_for_model_update()` when the worker adopts new inference weights, between completed cycles, then resume `search.cycle(&mut new_model)`. It clears the arena, pending evaluation bookkeeping, private-root statistics, simulation counters, paths and waiters. Current game states, game IDs, occupied/game-over flags and RNG state are preserved. Active roots are reevaluated by the next cycles. The call is idempotent and retains arena allocation capacity for reuse. If used to abort a manually collected trace, discard its old snapshots/results too. This API does not load weights or schedule model updates; the caller does that.

PUCT stores visits/totals as u64 and Q as f64, updating Q with `q += (value-q)/next_visit_count`. Selection and target normalization compute in f64. Checked integer increments fail explicitly at the theoretical u64 limit instead of wrapping; counts are not saturated or rescaled.

EXP3 stores log weights and strategy accumulators as f64. Mixed probabilities, sampling and stored Choice probabilities are also f64, keeping backup probabilities consistent with selection. When an updated action's stored log weight is at least 1024, its player's finite weights are shifted by their common maximum before the next update. This preserves the mathematical policy, leaves illegal-action -infinity masks intact, and avoids scanning weights on every backup. Strategy sums are never rescaled. Model priors/values and emitted training-policy arrays remain f32.

Per-node game IDs increased from 16 to 64; the existing u8 count/cursor can represent 64. The 65th distinct claim still replaces the oldest claim. Sweep-age behavior is unchanged.

| Node variant | Previous bytes | New bytes |
|---|---:|---:|
| PUCT | 2168 | 3552 |
| EXP3 | 1520 | 2896 |

The larger ID array accounts for 96 bytes of the increase. At G=8192, S=100 the unchanged formula permits 6,553,600 nodes. PUCT node payload at that limit is 23.28 GB. Allow approximately 31 GB for a fully grown arena including current Vec rounding, index and free-list allowance, or 45 GB transiently during node-vector growth. Roots, paths and other worker/model allocations are additional. This remains within the proposed 100 GB budget. These figures supersede earlier node-memory estimates. Wider arithmetic and more retained claims can change search trajectories and occupancy; previous traces are historical measurements, not guarantees for this version.

Validation: 21 relevant tests pass in both debug and release (12 variant, 5 numerical/ID capacity, 2 flush, 2 sweep lifetime). Flush tests cover settled and in-flight work, reset idempotence, RNG and game preservation, root reevaluation and resumed stepping for PUCT and EXP3. Late Q updates remain nonzero at 100 billion visits. A 20-million-backup probe retains constant f32(0.6) exactly and gives the expected mean for alternating f32(0.4)/f32(0.6). See fp-probe-hardened.csv; its exp3_* rows intentionally retain the old f32 arithmetic demonstrations, whereas its PUCT rows invoke the current production implementation. Both release binaries build and both variants complete a small profiler smoke run.

The full integration suite is still blocked by old tests referencing removed APIs (diag, IdWrite, shifted, probe_lengths, overflows and last_depth). Variant tests were updated to the current precision and to verify crossing the old u32 limit; obsolete saturation-field assertions were replaced. No removed diagnostics were reintroduced.
