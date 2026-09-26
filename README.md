# alpha-lines

KLENT self-play training for alpha-lines. The Rust engine runs the game and collects positions; PyTorch models predict a spatial policy and an action value for each playable square.

## Train locally

Install dependencies with `uv sync`, build the native library, then start training:

```bash
cargo build --release --manifest-path rust/Cargo.toml
./scripts/train_local.sh --device cuda
```

For a CPU smoke run, use `--device cpu --no-compile`. The default configuration is in `config.json`. `klent.cycles` is the number of additional cycles; zero runs until stopped. Checkpoints and logs are written under `checkpoints/klent/<run-id>/` and `logs/klent/<run-id>/`.

Resume with `--resume checkpoints/klent/<run-id>/cycle-XXXXXX.pt`. The resume checkpoint supplies the architecture, weights, optimizer state, completed cycle, and moving anchors. The preceding 64 cycle files, including `cycle-000000-weights.pt` when needed, must be beside it; training loads them into system RAM. If you resume into a new checkpoint directory, those files are linked or copied there. Current training settings come from `config.json`. Older checkpoints with auxiliary heads can still supply their policy, action-value, and tower weights; the removed head weights are ignored. The fixed strength-test opponent is selected by `klent.reference_checkpoint`.

The 1024-game arena has four equal partitions. Side 1 uses the current model throughout. Side 2 uses the current model without uniform exploration in the first partition, the current model with uniform exploration in the second, then two historical checkpoints without uniform exploration. Side 1 mirrors the exploration setting of each partition. Each partition balances player 0 and player 1 assignments. The historical checkpoints are `d` and `2d` cycles old, where `d` is the largest power of two at most half the current cycle, capped at 32. The current model generates policy and value targets for both perspectives; historical models only select moves. KLENT collects improved-policy targets and discounted action-value returns. `klent.policy_recalculation` optionally recomputes improved-policy targets for each minibatch. The loss uses `policy_loss_weight` and `q_loss_weight`.

Each cycle tests the new model against the previous model, three score-gated anchors, and the fixed reference checkpoint. The thresholds are configured by `klent.anchor_thresholds`.

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
