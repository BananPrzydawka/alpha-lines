KLENT training:

```sh
PYTHONPATH=python uv run modal run -m klent.modal_train --cycles 5
```

Evaluation assigns the new model to P1 in the first half of the games and P0
in the second half. Results always refer to the new model. `test_games` must
be even. Policies are sampled with the existing legal-action masking.

Self-play times are arithmetic means over all arena steps. Minibatch times
are arithmetic means over all batches, including the padded final batch.
Losses are weighted by valid position counts. First-cycle timing still includes
compilation. The checkpoint also records these averages.

Each completed cycle saves `klent/<unique-run-id>/cycle-000001.pt` on the
`alphalines` Modal volume. The printed container path starts `/checkpoints/`.
Files contain model weights, optimizer state, model/training configuration,
cycle metrics and torch RNG states. Writes use a temporary file followed by
rename and explicit volume commit. Separate runs never overwrite one another.
Native arena RNG state is not saved, so resume starts fresh games rather than
replaying an interrupted run exactly.

Model-only training benchmark:

```sh
PYTHONPATH=python uv run modal run -m klent.modal_train::benchmark
```

Optional flags: `--model resnet`, `--batch-size 8192`, `--warmup 20`,
`--iterations 100`. Defaults use the configured model and training minibatch.
It uses BF16, channels-last, max-autotune compilation, cuDNN benchmarking,
resident synthetic inputs/targets, and the same CE + played-action MSE and AdamW.
Warmup includes backward and optimizer initialization. CUDA events measure
forward plus loss, backward, optimizer and total; synchronization occurs after
the measurement loop. Reports include mean/median/best and wall time per step.
No Rust simulation, transfers, checkpoint I/O or per-batch printing is included.
This is a favorable measured reference, not a guarantee of the hardware's peak.
Compare steady-state training cycles, not the compilation-heavy first cycle.

A separate untimed diagnostic reports shared-trunk policy/Q gradient norms on
up to 32 synthetic examples. It does not establish balance on actual self-play
data. CE includes target entropy even when predictions match the target;
therefore its scalar magnitude is not directly comparable to MSE or gradient
strength. Both current losses average over the same valid perspectives; Q loss
is taken only at the played action, not averaged over 160 action outputs.

Historical evaluation keeps eight independent weight snapshots in CPU RAM.
After cycle C, opponents are C-1, C-2, C-4 and C-8 when available; cycle 0 is
the initial untrained model. Each matchup uses the full configured test_games,
with balanced player assignments. One compiled opponent model is reused by
copying in the selected weights. No historical weights are read from the volume.
The table reports current-model W/D/L and score (wins + half draws) / games.
All matchup results and opponent cycle numbers are included in saved metrics;
legacy top-level W/D/L fields refer to the immediately previous model.
History lasts for the current process only.

Local GPU / SSH training (Ubuntu GPU image with CUDA 13-compatible driver):

```sh
./scripts/setup_gpu.sh
./scripts/train_local.sh --resume checkpoints/cycle-000009.pt --cycles 20
```

`--cycles 20` means 20 additional cycles (10 through 29 in this example).
Omit `--resume` to start fresh. `--cycles 0` runs until stopped, with no Modal
timeout. The setup script installs uv, Rust and locked Python dependencies,
builds the native library and checks GPU availability. It does not change the
NVIDIA driver or profiler permissions and does not reboot.

Resume restores checkpoint model/training configuration, weights, optimizer,
torch RNG and cycle number. Saved configuration takes precedence over config.json
for that run. Opponent history starts with the resumed model and builds up again.
Local output defaults to a new `checkpoints/klent/<run-id>` directory; override
with `--checkpoint-dir PATH`. Existing checkpoint files are never overwritten.
Run inside tmux so disconnecting SSH does not stop training. Download saved files
before releasing the rented storage.
