# Numerical risks over 24–72 hours

This analysis uses the completed small-configuration instrumentation and previous throughput measurements. It does not use the additional deployment-scale numerical check; extrapolation is explicitly a stress scenario, not a measured high-scale visit distribution. Search arithmetic is unchanged.

## Measured traffic and counts

The earlier G=1024/B=T=512 PUCT run completed 11,421,300 simulations associated with emitted moves in 103.764 s, at least 110,070 simulations/s. The earlier G=8192/B=T=2048 run completed 12,090,300 in 106.237 s, at least 113,805/s. These omit partially completed searches at the final boundary. EXP3 ran at least 119,183 simulations/s. Simulations are not model rows: duplicate waiters can back up from one evaluation. A simulation can update several nodes but cannot visit the same position twice on its monotonic game path.

Use approximately 114,000 simulations/s as a PUCT ballpark: 9.83 billion in 24 hours and 29.50 billion in 72 hours. These are CPU-stub wall-clock rates, not real network inference throughput. All time estimates scale inversely with actual throughput and apply to the interval without a model-update flush.

Across all 200 instrumented small-configuration steps, including private roots:

| Statistic | Maximum |
|---|---:|
| PUCT total visits per player | 18,430 |
| PUCT visits on one edge | 11,073 |
| EXP3 positive log weight | 19.6236 |
| EXP3 one-action strategy accumulator | 900.526 |

Instrumentation scans immediately before each sweep and records arena and roots separately. These are boundary-observed maxima; EXP3 weights can be overwritten by evaluation, so this is not a per-operation maximum detector. PUCT counts are monotone within each node lifetime. PUCT numerical run occupancy matches all 200 previous steps; EXP3 matches the first 50 previously measured steps. Runs with extra numerical scans are not used for throughput estimates.

PUCT maxima occurred early (total at step 2, edge at step 49) and did not grow monotonically across the run. A node surviving 199 sweeps does not imply that it accumulated 199 sweeps of heavy traffic.

An aggressive simple deployment scaling is 8x the per-node visit counts, reflecting 8x G despite only 4x B/T: approximately 147,440 total visits and 88,584 edge visits. This is a stress estimate, not a mathematical upper bound. Sharing, retention and search trajectories can scale nonlinearly. At those counts, integer overflow and complete Q-update freezing are distant, although current f32 arithmetic already has measurable rounding bias.

## PUCT floating point

The production update is `(q * n_as_f32 + value) / (n_as_f32 + 1.0)`.

Binary32 has 24 significant binary bits. At q=0.5 the next larger representable value is 2^-24 (approximately 5.96e-8) away. A positive change below half that can disappear. The desired update is `(value-q)/(n+1)`, so with residual 0.1 this becomes an issue at a few million edge visits. This depends on q, residual and direction; it is not one universal visit cutoff.

Production-backup probes, q=0.5 and value=f32(0.6):

| Existing edge visits | Ideal Q change | Actual Q change |
|---|---:|---:|
| 100,000 | 9.9999e-7 | 1.01328e-6 |
| 1,000,000 | 9.99999e-8 | 1.19209e-7 |
| 4,194,304 | 2.38419e-8 | 0 |
| 16,777,216 | 5.96047e-9 | 5.96046e-8 |
| 100,000,000 | approximately 1e-9 | 0 |

At 2^24 the expression `n_as_f32 + 1.0` can equal `n_as_f32`. Larger integer counts also cease to be individually representable. Errors affect both numerator and denominator; behavior is not merely slower learning.

Repeated production backups using the *same* f32(0.6) value, whose exact arithmetic mean is 0.60000002384:

| Backups | Stored Q |
|---|---:|
| 10,000 | 0.6000511 |
| 100,000 | 0.6004649 |
| 1,000,000 | 0.6006145 |
| 4,194,304 | 0.5912783 |
| 16,777,216 | 0.5964913 |
| 20,000,000 | 0.7090582 |

Alternating f32(0.4)/f32(0.6), exact mean approximately 0.5000000149, gives Q=0.5361425 after 20 million updates. These deterministic probes demonstrate accumulated numerical error without model sampling noise. They are adversarial arithmetic examples, not claims that the measured stub exhibits those Q errors.

A stable update `q += (value-q)/(n+1)` removes the repeated weighted-product/division problem; storing q as f64 is also needed to retain small late updates. Merely widening counts does not address this precision loss. Priors can remain f32. Computing selection scores in f64 as well avoids rounding away very small Q differences at selection.

## Runtime implications

Using 113,805 simulations/s and a persistent edge receiving the stated share:

| Share of simulations | Edge updates/s | Time to 2^22 visits | Time to 20 million visits |
|---|---:|---:|---:|
| 0.01% | 11.4 | 102.4 h | 488.2 h |
| 0.1% | 113.8 | 10.24 h | 48.82 h |
| 1% | 1,138 | 1.02 h | 4.88 h |

To reach 2^22 edge visits requires an average of 48.5 updates/s over 24h, or 16.2/s over 72h. To reach 2^24 requires 194.2/s or 64.7/s respectively. Small fractions of worker traffic can therefore expose float issues within the requested window if the same edge persists and remains active.

As another deliberately pessimistic stress scenario, suppose the 8x-scaled edge maximum (88,584) accumulated anew every approximately 104 seconds into the same persistent edge. That is roughly 850 updates/s: several million visits in approximately 1.4 h, 20 million in approximately 6.5 h. This extrapolation contradicts neither the arithmetic nor throughput budget, but it is not the observed plateau/recycling behavior and must not be called a prediction.

## Integer overflow

The unchecked u32 total/edge increments overflow at 4,294,967,295. At least 49,710 updates/s to the same node/player would reach it in 24 h; 16,570/s would reach it in 72 h. These are approximately 43.7% and 14.6% of the measured deployment simulation throughput. Total can overflow while individual edges remain below the limit because total sums all actions. Debug builds panic; release increments wrap, invalidating cached totals, exploration and averaging.

Under the approximately 850 edge-updates/s stress case, 72h gives approximately 220 million edge visits, well below u32 overflow but already far into the f32 failure regime. u64 provides comfortable integer headroom without changing count semantics; saturating totals independently of edges would break invariants.

## EXP3 floating point

EXP3 uses gamma=0.1 and at most 80 legal actions. Its probability floor is gamma/K, so a valid backup increment `gamma * reward / (p*K)` is at most 1 for reward in [0,1]. Strategy sums also add at most 1 per action per selection. Thus even billions of updates are nowhere near f32's approximately 3.4e38 finite maximum. The issue is spacing between finite floats.

For K=80, reward=0.5, p=0.9, the intended log-weight increment is approximately 0.0006944445:

| Stored log weight | Actual f32 increment |
|---|---:|
| 1,024 | 0.0007324219 |
| 8,192 | 0.0009765625 |
| 16,384 | 0 |

At equal conditional reward 0.5, each action's expected log-weight increase per node selection is gamma*0.5/K=0.000625, independent of its sampling probability. Without recentering, a common offset of 16,384 takes on the order of 26 million node selections. At 114 selections/s that is about 64 hours; substantial quantization appears earlier. Actual rewards/legal counts vary. The current softmax subtracts the maximum only in a temporary expression; it does not recenter the stored log weights. Subtracting a common maximum from stored finite weights preserves the mathematical strategy and prevents this avoidable common-offset growth.

For strategy_sum, repeated addition of fixed probabilities stalls in direct f32 probes:

| Probability added | First unchanged addition | Frozen accumulator |
|---|---:|---:|
| 0.9 | 17,057,332 | 16,777,216 |
| 0.1 | 18,073,721 | 2,097,152 |
| 0.0125 | 18,073,721 | 262,144 |
| 0.00125 | 22,918,664 | 32,768 |

At approximately 119 selections/s to one EXP3 node (0.1% of measured worker throughput), this is approximately 40–54 hours. Distortion precedes complete stalling, and changing policies can make small new contributions disappear much earlier in an accumulator built under a larger old probability. Recentring log weights does not repair strategy_sum; use f64 for that accumulator. Halving strategy sums preserves the target immediately but changes the weight of future observations, so that is an algorithmic aging choice rather than a numerically neutral fix.

The observed EXP3 maxima (19.6 and 900.5) are far below these examples. Even a crude 8x extrapolation does not reach log freezing or large-probability strategy-sum freezing, but that does not bound accumulation over 72 hours.

## Recommendation

Flush the entire search state when an actor adopts a new model version, preserving game positions but resetting arena, private-root statistics, pending work and sim counts. This prevents stale priors/values from contaminating new-model searches. Arena::clear alone is insufficient. No evidence from this analysis justifies a fixed hourly flush solely for arithmetic safety: rates depend on actual node traffic, not age or process uptime.

Also harden arithmetic: u64 PUCT counts, f64 Q with the stable difference-form mean, f64 EXP3 strategy accumulation and stored-log recentering (f64 log weights are cheap additional margin). Keep priors/model outputs f32. This is justified by plausible hot-node traffic over 24–72h, even though the observed representative traces are comfortably below severe failure thresholds. Prefer monitoring maximum per-node/per-edge counts and float magnitudes over using sweep age as a numerical alarm.

No production arithmetic or flush behavior was changed by this audit. Added optional profiler numerical output and reproducible probe artifacts. Raw files: puct-numerics-200.csv, exp3-numerics-200.csv, fp-probe.csv; probe source: fp_probe.rs. The existing full-suite API mismatch remains outside this analysis.
