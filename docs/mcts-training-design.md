# MCTS continuation from KLENT cycle 349

## Goal and baseline

Load `checkpoints/resume.pt` as the fixed KLENT baseline. The checkpoint reports
cycle 349 and contains FP32 KataGo weights and AdamW state. The MCTS run has its
own output directory. The baseline is never overwritten by run output.

Train only the existing policy and action value heads. The model still computes
its other heads, but their loss weights are zero. Evaluate new models with the
current KLENT direct-policy match procedure, using balanced sides and separate
matches against the baseline, the directly previous model, and two later fixed
anchors. Capture the later anchors after configurable numbers of MCTS cycles
(proposed defaults: 8 and 16). If an anchor equals the previous model, run that
match once and label it as both.

## Search and collection

Use the historical PUCT search mechanics, with the later numerical hardening and
capacity calibration preserved through commit `ddb1780`: a persistent arena of `g` game slots,
an evaluation buffer of `b` unique positions, a ready-to-step threshold `t`,
shared Zobrist-addressed nodes, private root statistics and noise per game,
node sweep after steps, and reseeding of terminal games. `g`, `b`, `t` and the
simulation count `s` are configurable. Derive node capacity by default from the
later empirically tested formula `g * s * 8` (roughly four times the largest
50-step measured peak); expose the factor in config and permit an explicit
capacity override. The earlier `g * s / 2 * 5 / 4` rule was disproven by the
arena calibration in commit `fbb2f56`. This capacity limits live
search nodes and is independent of the `t * steps` training-target buffer.
Check that `t <= g` and `b <= g`.

At each model call, run a `2*b` model batch: two perspectives per position and
zero-filled rows for any shortfall. Encode each queued position with the same
five planes and playable-square ordering used by KLENT. Legal softmax probabilities
from the model policy head become the priors and weight the model action values
into one state value per player. Search backs up those values and uses exact game outcomes at
terminal leaves. A step emits the pre-move board, visit-count policy for both
players, backed-up action values for visited moves, and the played actions. Unvisited
moves have no action value target and make no contribution to its loss. The
`q_visit_weighting` option selects equal weighting (default) or weight proportional
to each move's visit count. Retain these records
in a per-game history until the game ends, even though this experiment's
targets are usable immediately. Store a separate training copy of each emitted
record before advancing the game.

One collection cycle consists of `steps` search steps. An individual step moves
exactly the first `t` ready games; any others remain ready for a later step.
The cycle therefore holds `t * steps` source positions. Keep game positions and their
per-game histories across training; flush all shared nodes and root statistics
after each network update, then reevaluate roots with the new model.

Shuffle the collected positions and train for one epoch per collection cycle.
Augmentation is applied to board planes and both policy and value targets with
the same spatial transform. Configure `double_flip_augment` and
`all_flip_augment` as boolean toggles. Both off is the default; double produces
two examples per position; all produces four and takes precedence when both
toggles are enabled. Apply augmentation before shuffling so the variants can land
in different minibatches. The last minibatch may be smaller than the configured
size; only actual rows contribute to the loss.

## Metrics and outputs

Report raw policy cross-entropy, raw action value MSE, their configured weights,
and total loss. Record search wall time, search CPU time, search model time,
training preparation time, training model time, evaluation time, checkpoint/log
time, and total wall time. Compile and warm the model's search, training, and
evaluation shapes before the first timed cycle, and report that initial setup
separately. Report model calls, evaluated positions, model batch
fill, node allocation failures, steps, source positions, augmented positions and
minibatches. Model batch fill is evaluated positions divided by `b * model calls`;
it is distinct from search node capacity.
Initialize a fresh AdamW optimizer from the configured learning rate and decay.
Save an atomic checkpoint after each complete cycle with model, optimizer,
MCTS-cycle count, source KLENT cycle, effective configuration and RNG state.
Write one JSON metrics row per cycle and a compact console report.

## Confirmed choices

The selected configuration is: MCTS cycles start at 1; double augmentation is
identity plus 180° rotation; all augmentation uses identity, vertical, horizontal,
and 180° transforms; player identities remain unchanged by augmentation.

The original KLENT baseline is carried in MCTS checkpoints so it remains the
reference opponent after resuming MCTS training.
