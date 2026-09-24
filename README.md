# Alpha-lines

KLENT self-play training with a Rust game engine and a PyTorch KataGo model.
Local/Verda training and Modal are supported.

## Checkpoints and logs

- `checkpoints/resume.pt`: the **manually selected FP32 input**, the only checkpoint tracked by Git. Move or copy your chosen checkpoint here before committing. It is never automatically selected or replaced.
- `checkpoints/klent/<run-id>/cycle-XXXXXX.pt`: generated checkpoints, kept locally and ignored by Git.
- `checkpoints/klent/<run-id>/anchors/`: at most three standalone evaluation models, transferred once per run.
- `checkpoints/klent/bf16/`: older BF16 checkpoints, kept locally and excluded from result downloads.
- `logs/klent/<run-id>/`: `console.log` (Verda), `run.json` and `metrics.jsonl`, ignored by Git.

Each invocation creates a separate run. Repeating the Verda command starts another
experiment from the same `resume.pt`, even when newer outputs exist.
To change the starting point, replace `resume.pt` yourself. For example:

```sh
cp checkpoints/klent/RUN_ID/cycle-XXXXXX.pt checkpoints/resume.pt
mkdir -p checkpoints/anchors
cp checkpoints/klent/RUN_ID/anchors/anchor-*.pt checkpoints/anchors/
```

The copied `anchors/` directory must travel with a slim `resume.pt`; it is ignored
by Git. On a new Verda box, transfer `checkpoints/anchors/` once before training.
You can also resume directly from a run checkpoint in its original directory,
or pass `--anchor-dir` to point to a directory containing `anchors/`.
Older checkpoints with embedded anchors remain readable and are split into
standalone anchor files when training next saves a checkpoint.

## Verda workflow

Edit `config.json` and select `checkpoints/resume.pt`, then publish locally:

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

The launcher installs dependencies, builds Rust, and trains in the foreground.
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
architecture, FP32 weights, AdamW state and completed cycle number. Only KataGo
checkpoints are supported.
An unchanged seed restores torch RNG state; changing it reseeds torch.
BF16-only legacy checkpoints are unsupported. Training retains BF16 autocast
with FP32 weights, optimizer moments and losses.

Snapshots are saved atomically after complete cycles. Native games and opponent
history are rebuilt on resume, so restarting is not an exact replay.
Evaluation compares each completed cycle against the immediately previous model and
up to three fixed anchor models, saved at completed cycles divisible by eight.
Once three anchors are collected, the panel stays fixed. Anchors are stored in each
run's `anchors/` directory as three separate model files. Cycle checkpoints contain
only their anchor cycle numbers. The first cycle after a restart compares against
the resumed model. If the resumed checkpoint
has no anchors, its model becomes the first anchor immediately. Later anchors are
captured eight cycles apart from that first anchor. When an anchor is also the previous model,
that matchup runs only once and is labeled as both.
`test_games` must be even and applies to each opponent separately.
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
set output paths. KataGo model settings are in `config.json`.

## Modal

```sh
PYTHONPATH=python uv run modal run -m klent.modal_train
```

This starts a fresh run using the configured Modal GPU and timeout. Add `--smoke`
for small batches. Checkpoints and metrics persist under
`/checkpoints/klent/<run-id>/` on the `alphalines` Modal volume; each completed
checkpoint is committed to the volume.

## Project and checks

- `python/klent/`: training, native bindings, checkpointing, logging and Modal entrypoint.
- `python/models/`: KataGo and shared layers.
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
