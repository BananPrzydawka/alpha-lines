# Alpha-lines

KLENT self-play training with a Rust game engine and PyTorch KataGo/ResNet models.
Local/Verda training and Modal are supported.

## Checkpoints and logs

- `checkpoints/resume.pt`: the **manually selected FP32 input**, the only checkpoint tracked by Git. Move or copy your chosen checkpoint here before committing. It is never automatically selected or replaced.
- `checkpoints/klent/<run-id>/cycle-XXXXXX.pt`: generated checkpoints, kept locally and ignored by Git.
- `checkpoints/klent/bf16/`: older BF16 checkpoints, kept locally and excluded from result downloads.
- `logs/klent/<run-id>/`: `console.log` (Verda), `run.json` and `metrics.jsonl`, ignored by Git.

Each invocation creates a separate run. Repeating the Verda command starts another
experiment from the same `resume.pt`, even when newer outputs exist.
To change the starting point, replace `resume.pt` yourself. For example:

```sh
cp checkpoints/klent/RUN_ID/cycle-XXXXXX.pt checkpoints/resume.pt
```

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
type, architecture, FP32 weights, AdamW state and completed cycle number.
An unchanged seed restores torch RNG state; changing it reseeds torch.
BF16-only legacy checkpoints are unsupported. Training retains BF16 autocast
with FP32 weights, optimizer moments and losses.

Snapshots are saved atomically after complete cycles. Native games and opponent
history are rebuilt on resume, so restarting is not an exact replay.
Evaluation retains 32 CPU weight snapshots in RAM and compares against snapshots
aged 1, 2, 4, 8, 16 and 32 cycles when available, with balanced player assignments.
`test_games` must be even and applies to each opponent separately.
Metrics record losses, W/D/L, historical matchups, counts and timings; `run.json`
records the effective configuration, precision, hardware and resume source.
Completed positions are shuffled before each cycle's single training epoch. Shuffle
time is reported as `shuffle_seconds`; expansion of mark classes into training
planes is reported as `scoring_head_processing_seconds`.

The auxiliary mark head predicts six classes on occupied squares: own or opponent
mark contributing 0, 1, or 2 points. Its target is derived from the current
border-connected diagonal runs and stored with each self-play position. The
head is added automatically when resuming a two-head FP32 checkpoint, preserving
existing weights and AdamW moments. To compile out mark classification and its
storage, set `TRACK_MARK_CLASSES` to `false` in `rust/src/game.rs` and rebuild Rust;
training detects this and omits the head and auxiliary loss.

After dependency setup, local training can also run directly:

```sh
./scripts/train_local.sh --resume checkpoints/resume.pt
```

Omit `--resume` to train from scratch. Optional `--checkpoint-dir` and `--log-dir`
set output paths. KataGo and ResNet have independent model settings in `config.json`.

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
- `python/models/`: KataGo/ResNet and shared layers.
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
