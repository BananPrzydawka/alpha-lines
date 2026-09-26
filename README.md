# alpha-lines

KLENT self-play training for alpha-lines. The Rust engine runs the game and collects positions; PyTorch models predict a spatial policy and an action value for each playable square.

## Train locally

Install dependencies with `uv sync`, build the native library, then start training:

```bash
cargo build --release --manifest-path rust/Cargo.toml
./scripts/train_local.sh --device cuda
```

For a CPU smoke run, use `--device cpu --no-compile`. The default configuration is in `config.json`. `klent.cycles` is the number of additional cycles; zero runs until stopped. Checkpoints and logs are written under `checkpoints/klent/<run-id>/` and `logs/klent/<run-id>/`.

Resume with `--resume checkpoints/resume.pt`. This compact cycle-55 checkpoint supplies the architecture, core model weights, optimizer state, and completed cycle. Its moving anchors were removed to keep the file below GitHub's size limit; on resume, all three anchors start from the cycle-55 model. Current training settings come from `config.json`. Older checkpoints with auxiliary heads can still supply their policy, action-value, and tower weights; the removed head weights are ignored. The fixed strength-test opponent is selected by `klent.reference_checkpoint`, which is a separate checkpoint file.

KLENT collects improved-policy targets and discounted action-value returns. `klent.policy_recalculation` optionally recomputes improved-policy targets for each minibatch. The loss uses `policy_loss_weight` and `q_loss_weight`.

Set `klent.gradient_accumulation` to `true` to average gradients over every position in a cycle and take one AdamW step after all minibatches. The final partial minibatch is weighted by its valid positions. `false` keeps one optimizer step per minibatch. The setting can be changed when resuming; the checkpoint's optimizer state is retained.

Each cycle tests the new model against the previous model, three score-gated anchors, and the fixed reference checkpoint. The thresholds are configured by `klent.anchor_thresholds`.

To compare training-gradient directions from the cycle-55 `checkpoints/resume.pt` without updating its weights, run:

```bash
PYTHONPATH=python uv run --no-sync modal run -m klent.modal_train::gradient_comparison
```

The diagnostic collects two independently seeded arenas. It reports the mean gradient dot product, cosine similarity, and fraction of positive dot products between the arenas for effective batches of 2,048 through 32,768 perspectives. `--chunks 8` shortens the run. It loads only the current policy and action-value heads from the checkpoint; the source checkpoint is untouched.

## GPU and Modal

`./scripts/train_verda.sh` launches training on the GPU machine. `./scripts/pull_verda.sh --ip BOX_IP` downloads checkpoints and logs. To run one cycle on Modal:

```bash
PYTHONPATH=python uv run modal run -m klent.modal_train
```

## Validate

```bash
cargo test --manifest-path rust/Cargo.toml
PYTHONPATH=python uv run --no-sync python -m unittest klent.test_klent models.test_models
```

The model implementations live in `python/models/`, KLENT training and native bindings in `python/klent/`, and the Rust game engine and collection arena in `rust/src/`.
