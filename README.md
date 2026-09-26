# Alpha-lines

KLENT self-play training with a Rust game engine and selectable PyTorch models.
Local/Verda training and Modal are supported.

## MCTS continuation

MCTS training starts from the FP32 KLENT checkpoint at `checkpoints/resume.pt`
(currently cycle 349). It creates a separate MCTS run, numbers its first cycle 1,
and starts a fresh AdamW optimizer. Configure it under `mcts` in `config.json`.

```sh
./scripts/train_mcts_local.sh --resume checkpoints/resume.pt
```

For Verda, publish the checkpoint and config, then run
`./scripts/train_mcts_verda.sh` on the GPU machine. `pull_verda.sh` downloads
MCTS checkpoints and logs along with KLENT outputs. MCTS output lives under
`checkpoints/mcts/<run-id>/` and `logs/mcts/<run-id>/`; it never replaces
`checkpoints/resume.pt`. Resuming an MCTS cycle checkpoint restores its optimizer
and MCTS cycle number. Every MCTS checkpoint also carries the original KLENT
baseline weights for strength tests.

The Rust PUCT search uses `g` persistent game slots, evaluates at most `b`
positions per model call, and steps the first `t` ready games. Every search
model call runs a `2*b` batch: both player perspectives for each board, with
zero-filled rows when fewer than `b` states are available. It repeats this
for `steps` search steps per training cycle, collecting exactly `t * steps`
training positions. Search-node capacity is a separate limit. It defaults to
`g * s * node_capacity_factor`, using the later calibrated factor of 8. A
50-step grounded-model run peaked at about `2 * g * s` nodes, so this budget was
roughly four times its measured peak; it is not a worst-case bound. With the
current `g=2048` and `s=100`, the factor gives 1,638,400 nodes. To override
the budget directly, add `node_capacity` under `mcts` in `config.json`.
The legal softmax of the model policy head supplies
search priors and weights the action value head into leaf state values. Terminal leaves use actual
game results. Training uses the search visit distribution and the backed-up mean
return for each visited action; unvisited actions carry zero value-loss weight.
`q_visit_weighting` is `equal` by default or `visits` to weight value errors by
visit count. Search nodes are cleared after each model update while games and
their histories continue.

`double_flip_augment` and `all_flip_augment` are independent boolean config
toggles. Both off means identity only. Double uses identity and 180° rotation.
All uses identity, vertical flip, horizontal flip, and 180° rotation. If both
toggles are on, the all-flips set applies once.
The same transform is applied to boards and both targets, without swapping
players. Single-axis flips change the playable-square parity; they are an
experimental augmentation. On the opening position, 180° rotation also changes
which half is available to each player; this is accepted for training data.

After each cycle, the new model plays the same direct-policy match used by KLENT
against the original KLENT model, the directly previous MCTS model, and anchors
captured after the two configured `anchor_after` offsets (8 and 16 by default,
measured from the start of each run).
The report and JSON metrics include both losses, model batch fill, node counts,
and separate search, training, evaluation, and checkpoint times. Compiled search,
training, and evaluation graphs are warmed before cycle timing; the one-time
compile/warmup time appears under initial setup. Each MCTS step prints its own
wall time and an updated collection-time estimate. More detail is
in [the MCTS design](docs/mcts-training-design.md).

Model batch fill measures how many of the `b` evaluation slots were used per
search call. Node allocation failures count searches that tried to create a
node after reaching node capacity. They measure different resources.

## Checkpoints and logs

- `checkpoints/resume.pt`: the manually selected FP32 training input. Move or copy your chosen checkpoint here before committing. It is never automatically selected or replaced.
- `checkpoints/349.pt`: the fixed KLENT strength-test opponent. Keep this file available for local, Modal, and Verda training.
- `checkpoints/klent/<run-id>/cycle-XXXXXX.pt`: generated checkpoints, kept locally and ignored by Git.
- KLENT cycle checkpoints contain all three active moving anchors, so a single checkpoint can be copied to `resume.pt` without losing them.
- `checkpoints/klent/bf16/`: older BF16 checkpoints, kept locally and excluded from result downloads.
- `logs/klent/<run-id>/`: `console.log` (Verda), `run.json` and `metrics.jsonl`, ignored by Git.

Each invocation creates a separate run. The plain Verda command starts from
scratch; `--resume` starts from the manually selected `resume.pt` even when
newer outputs exist. To change the resume starting point, replace `resume.pt`
yourself. For example:

```sh
cp checkpoints/klent/RUN_ID/cycle-XXXXXX.pt checkpoints/resume.pt
```

Score-gated anchors in a new KLENT checkpoint are restored from `resume.pt`.
Older fixed-duration anchor formats are ignored; when resuming one of those
checkpoints, the input model initializes all three moving anchors.

## Verda workflow

Edit `config.json`, include `checkpoints/349.pt`, and publish locally:

```sh
git add .
git commit -m "commit"
git push origin HEAD:main
```

On the rented Ubuntu GPU machine (with a working CUDA 13-compatible NVIDIA driver):

```sh
apt-get update &&
apt-get install -y git &&
git clone https://github.com/BananPrzydawka/alpha-lines ~/projects/alpha-lines &&
cd ~/projects/alpha-lines &&
./scripts/train_verda.sh
```

The launcher starts a fresh KLENT model by default, installs dependencies,
builds Rust, and trains in the foreground. Use `./scripts/train_verda.sh --resume`
only when you want to continue from `checkpoints/resume.pt`.
Keep SSH connected. On an existing checkout, run `git pull --ff-only origin main`
before launching a new experiment.

Download results **from your local project directory**, before deleting the box:

```sh
./scripts/pull_verda.sh --ip BOX_IP
```

This merges runs directly into local `checkpoints/klent/` and `logs/`, excludes the
remote input checkpoint, and never replaces your selected `resume.pt`. Rerun to
continue a transfer. The script works from any working directory, skips unfinished
checkpoint files and the local BF16 archive, and does not delete local files.
It connects as `root` to `/root/projects/alpha-lines/` on the box.

## Training configuration and resume

`klent.cycles` is the number of **additional** cycles (0 runs until stopped).
Current `klent` settings control self-play, buffer size, minibatches, evaluation,
learning rate, weight decay, alpha/beta/lambda and seed. Checkpoints supply model
architecture, FP32 weights, AdamW state and completed cycle number. KLENT supports
`katago`, `katago_tf`, `resnet`, and `maia`.
An unchanged seed restores torch RNG state; changing it reseeds torch.
BF16-only legacy checkpoints are unsupported. Training retains BF16 autocast
with FP32 weights, optimizer moments and losses.
`klent.policy_recalculation` defaults to `false`. When enabled, each training
minibatch recomputes KLENT's legal improved-policy target from that minibatch's
current policy and action-value outputs, using the configured `alpha` and `beta`.
The target is detached before policy and opponent-policy losses; value training
still uses the stored returns. The opening left/right move rule is recovered
from the board and paired player rows.

Snapshots are saved atomically after complete cycles. Native games and opponent
history are rebuilt on resume, so restarting is not an exact replay.
Evaluation compares each completed cycle, in order, with the immediately previous
model, three moving anchors, and the fixed `checkpoints/349.pt` model. A fresh run
initializes all three anchors from its first completed checkpoint (cycle 1), so
cycle 1 tests only the previous model and checkpoint 349. Anchor 1 advances to
the current model when its score exceeds 70%; anchors 2 and 3 use 80% and 90%.
These thresholds are configured by `klent.anchor_thresholds` and use wins plus
half of draws. Equality does not advance an anchor. Once initialized, anchors
are stored in every cycle checkpoint and continue across resume. Match results
are reused when multiple anchor slots hold the same checkpoint, while each slot
still has its own report row. `test_games` must be even.
Metrics record losses, W/D/L, historical matchups, counts and timings; `run.json`
records the effective configuration, precision, hardware and resume source.
Completed positions are shuffled before each cycle's single training epoch.
The cycle report shows total time, CPU time (including shuffle and mark-target
preparation), self-play model time, training time, evaluation time, and other
time (including checkpoint and metrics writing). The total ends just before
printing the report, so the report's own print time is excluded. Detailed
shuffle and mark-target times remain in metrics. One-time initialization and
checkpoint loading appear separately as `Setup`.

The auxiliary mark head predicts six classes on occupied squares: own or opponent
mark contributing 0, 1, or 2 points. Its target is derived from the current
border-connected diagonal runs and stored with each self-play position. The
head is added automatically when resuming a two-head FP32 checkpoint, preserving
existing weights and AdamW moments. To compile out engine mark classification
and its storage, set `TRACK_MARK_CLASSES` to `false` in `rust/src/game.rs` and
rebuild Rust. The model still has mark heads; absent targets give them zero loss.

The immediate-score head predicts both current scores as 81-way logits (0–80),
ordered current player then opponent. It reads the final shared residual-tower
features alongside the policy, action-value and mark heads. The model receives
only board planes; scores stored with each self-play position are training
targets, not model inputs. Cross-entropy averaged over both players joins the
loss with `klent.immediate_score_weight`. The loss weights are configured
as `policy_loss_weight: 1.0`, `q_loss_weight: 2.0`,
`mark_class_loss_weight: 1.0`, and `immediate_score_weight: 1.0`.
The discounted-score head also outputs two 81-way score distributions from the
shared tower. Each completed game supplies soft targets by walking scores
backward from the final post-action score:
`D_t = (1 - discounted_score_lambda) onehot(score_t) + discounted_score_lambda D_(t+1)`.
The discount defaults to 0.94, independently of the Q-return `lambda`, and
`discounted_score_weight` defaults to 1.0. The target distributions are computed
before shuffling and stored with each position; this adds 648 bytes per buffered
position. Both score heads use cross-entropy and report their raw and weighted
losses separately. On resume from older FP32 checkpoints, missing heads are
added while existing weights and their AdamW moments are retained; the obsolete
score-input embedding, if present, is discarded.

After dependency setup, local training can also run directly:

```sh
./scripts/train_local.sh --resume checkpoints/resume.pt
```

Omit `--resume` to train from scratch. Optional `--checkpoint-dir` and `--log-dir`
set output paths. Choose `klent.model` in `config.json` and edit its corresponding
`<model>_model` section. `katago_tf` keeps KataGo's outer bottleneck and seven
heads, replacing both inner residual blocks with full-board transformer blocks.
`resnet` restores the historical squeeze-and-excitation tower and seven heads.
`maia` attends to the 80 playable squares and zero-fills the other 80 locations
in each spatial output; its two score heads still return full score distributions.
`maia_model.maia_big_version` defaults to `false`, using average-pooled GAB for
the paper's 3M/5M style. Set it to `true` for the learned GAB input projection:
each token is projected to 32 features, all 80 tokens are flattened, then mapped
to `gab_dim`. For the paper's 23M/79M widths, set `dim` to 512/1024 and
`gab_dim` to 128; the switch does not change those dimensions automatically.

## Modal

```sh
PYTHONPATH=python uv run modal run -m klent.modal_train
```

This starts a fresh run using the configured Modal GPU and timeout. Add `--smoke`
for small batches. Checkpoints and metrics persist under
`/checkpoints/klent/<run-id>/` on the `alphalines` Modal volume; each completed
checkpoint is committed to the volume.

For a single full MCTS cycle from the local `checkpoints/resume.pt`:

```sh
PYTHONPATH=python uv run modal run -m mcts.modal_train
```

This uses the `mcts` settings in `config.json`, overrides only `cycles` to 1,
and runs on the configured Modal GPU. The job allows at least two hours because
MCTS collection and strength testing can exceed the KLENT timeout. Its checkpoint
persists under `/checkpoints/mcts/<run-id>/`, with `run.json` and `metrics.jsonl`
under that run's `logs/` directory on the `alphalines` Modal volume.

## Project and checks

- `python/klent/`: training, native bindings, checkpointing, logging and Modal entrypoint.
- `python/models/`: KataGo, transformer KataGo, ResNet, Maia, and shared layers.
- `rust/src/`: game engine and KLENT arena.
- `rust/tests/`: independent game parity tests; KLENT unit tests also live in Rust source.
- `scripts/`: GPU setup, local/Verda launchers and result downloads.

```sh
cargo test --locked --manifest-path rust/Cargo.toml
cargo build --release --locked --manifest-path rust/Cargo.toml
PYTHONPATH=python uv run --no-sync python -m unittest klent.test_klent models.test_models
```

The Python suite exercises CPU training, checkpoint resume, current-setting
overrides, model contracts and historical evaluation without needing a GPU.
