# alpha-lines

KLENT self-play training for alpha-lines. The Rust engine runs the game and collects positions; PyTorch models predict a spatial policy and an action value for each playable square.

## Train locally

Install dependencies with `uv sync`, build the native library, then start training:

```bash
cargo build --release --manifest-path rust/Cargo.toml
./scripts/train_local.sh --device cuda
```

For a CPU smoke run, use `--device cpu --no-compile`. The default configuration is in `config.json`. `klent.cycles` is the number of additional cycles; zero runs until stopped. Checkpoints and logs are written under `checkpoints/klent/<run-id>/` and `logs/klent/<run-id>/`.

Resume with `--resume checkpoints/klent/<run-id>/cycle-000050.pt`. Checkpoints restore the model and optimizer. A resumed model becomes the first anchor at its saved cycle; earlier anchors are not restored. Interval-based fixed-opponent training likewise starts a new frozen opponent from the resumed model. Current training settings come from `config.json`.

KLENT collects improved-policy targets and discounted action-value returns. Training uses the stored targets and sums policy and action-value losses equally, with one AdamW step per minibatch.

Fresh training saves the untrained model as checkpoint and anchor cycle 0. `klent.anchor_interval` adds an anchor every specified number of cycles. On cycles divisible by `klent.test_interval`, strength tests compare the current model with the previous checkpoint, every existing anchor, and all optional reference checkpoints. New anchors are added after that cycle's tests. Skipped cycles still save a training checkpoint and record no strength-test result.

Set `klent.fixed_opponent` to `false` for normal self play, or `true` to play each block of `klent.opponent_interval` training cycles against a frozen copy of the model. The first opponent is cycle 0; after each block, the current model becomes the next opponent. In fixed-opponent mode, half the games use the trainable model as player 0 and half as player 1. `klent.both_sides=false` trains only on the trainable player's perspective. With `true`, the trainable model also generates targets for the frozen player's perspective while that player's moves still come from the frozen model.

Pass `--oponent checkpoints/349.pt` (or `--opponent`) to train against that exact checkpoint every cycle, regardless of `klent.opponent_interval`. This works with local training, Modal, and `train_verda.sh`. Pass the flag again on resume to keep that opponent; checkpoint files do not store its weights or path. Strength-test references remain controlled by `--reference`.

To compare training-gradient directions from the cycle-55 `checkpoints/resume.pt` without updating its weights, run:

```bash
PYTHONPATH=python uv run --no-sync modal run -m klent.modal_gradient
```

The diagnostic collects two independently seeded arenas. It reports the mean gradient dot product, cosine similarity, and fraction of positive dot products between the arenas for effective batches of 2,048 through 32,768 perspectives. `--chunks 8` shortens the run. It loads only the current policy and action-value heads from the checkpoint; the source checkpoint is untouched.

## GPU and Modal

`./scripts/train_verda.sh` launches training on the GPU machine. `./scripts/pull_verda.sh --ip BOX_IP` downloads checkpoints and logs. Modal starts fresh with no reference by default:

```bash
PYTHONPATH=python uv run modal run -m klent.modal_train::main
```

To resume and add multiple strength-test references, pass local checkpoint files or paths already on the mounted `/checkpoints` Modal volume. Local files are uploaded once for that run:

```bash
PYTHONPATH=python uv run modal run -m klent.modal_train::main \
  --resume /checkpoints/klent/<run-id>/cycle-000050.pt \
  --reference checkpoints/349.pt --reference checkpoints/another.pt
```

`klent.cycles` controls the number of additional cycles. Modal supports `--smoke` for a one-cycle small run. `./scripts/train_local.sh` and `./scripts/train_verda.sh` both accept `--resume MODEL_PATH`, `--oponent MODEL_PATH`, and repeated `--reference MODEL_PATH` flags. Verda paths must exist on that machine; Modal can also use paths on its mounted volume.

Use `--name RUN_NAME` with Modal, local training, or Verda to choose the folder name. For example, `--name baseline-349` writes to `checkpoints/klent/baseline-349/` and `logs/klent/baseline-349/`. The Modal launcher automatically downloads committed checkpoints and run logs into those local folders in the background while training continues. Keep the launching terminal open until the run finishes so the background download can complete; the final committed files are pulled once more when training ends.

Modal checkpoint writes run in a background thread in the same training container. Each cycle first copies the model, optimizer, and RNG state to CPU; the writer then serializes and commits that frozen snapshot while the next cycle trains. Anchor and opponent weights are not copied into checkpoints. The writer waits for the previous checkpoint before queuing another. Modal training targets a cycle-boundary stop 60 seconds before the function timeout, then waits for the last checkpoint to commit.

Set `klent.compile_model` to `true` or `false` to control `torch.compile`. Set `klent.compile_max_autotune` to `true` to use `torch.compile(mode="max-autotune")`; it defaults to `false` and has no effect when compilation is disabled.

## Checkpoint matchups

Run the standalone Modal matchup script with at least two checkpoints:

```bash
PYTHONPATH=python uv run modal run -m modal_matchups \
  --oponent checkpoints/349.pt \
  --oponent checkpoints/klent/<run-id>/cycle-000050.pt \
  --oponent checkpoints/klent/<run-id>/cycle-000100.pt
```

Repeat `--oponent` for any number of checkpoints (`--opponent` also works). Local files are uploaded temporarily; `/checkpoints/...` paths already on the Modal Volume work directly. `matchups.games` in `config.json` sets the even number of games per pair. Each pair plays half its games with each model as player 0, with all games for one model batched into one inference call per step. The script prints W/D/L results and a full score matrix, where each cell is the row model's wins plus half its draws against the column model. The `matchups` config also controls the seed and compilation settings.

## Validate

```bash
cargo test --manifest-path rust/Cargo.toml
PYTHONPATH=python uv run --no-sync python -m unittest klent.test_klent models.test_models
```

The model implementations live in `python/models/`, KLENT training and native bindings in `python/klent/`, and the Rust game engine and collection arena in `rust/src/`.
