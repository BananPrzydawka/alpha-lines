KLENT training:

```sh
PYTHONPATH=python uv run modal run -m klent.modal_train
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
It uses FP32 weights and AdamW state with BF16 autocast, channels-last, max-autotune compilation, cuDNN benchmarking,
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

Set `klent.cycles` in `config.json` before deploying (default: 20 additional
cycles; 0 runs until stopped). The cycle budget always comes from the current
config, even on resume. Starting from the supplied cycle 166, 20 cycles produces
cycles 167 through 186. Training settings come from the current `klent` config, including buffer and
arena sizes, minibatch size, evaluation games, learning rate, weight decay,
alpha/beta/lambda and seed. The current `m=1048576` therefore applies on resume.
The checkpoint supplies model type/architecture, weights, AdamW moments and cycle
number. Current learning rate and weight decay override saved optimizer settings.
An unchanged seed restores torch RNG state; a changed seed reseeds torch and the
native arena. Existing validation constraints still apply (`max_ply` must be 80,
and AdamW is the supported optimizer). FP32 parameters and AdamW moments are used, including when
resuming an older BF16 checkpoint; forwards use BF16 autocast and losses use FP32.
This cannot recover precision already lost during earlier BF16 training.

Commit and push these changes **and `checkpoints/cycle-000166.pt`** before creating
the box. Only that seed checkpoint is allowed through `.gitignore`; historical
and newly generated checkpoints stay ignored.

After connecting over SSH, clone the repository once:

```sh
apt-get update && apt-get install -y git &&
git clone https://github.com/BananPrzydawka/alpha-lines ~/projects/alpha-lines
```

Then run:

```sh
cd ~/projects/alpha-lines && ./scripts/train_verda.sh
```

For an existing checkout, update it first with `git pull --ff-only origin main`.
The launcher completes dependency setup, then finds the highest numbered
checkpoint beneath `checkpoints/` and trains directly in your terminal. Setup
failure or interruption stops the launcher before training starts. Let dependency
downloads finish. There is no tmux or background process; keep SSH connected.
Each run writes to unique directories under `checkpoints/klent/` and `logs/klent/`.
Rerun the same command to resume the latest saved cycle.

For foreground training use `./scripts/train_local.sh --resume
checkpoints/cycle-000166.pt` (on one line). Omit `--resume` for a fresh model.
Optional `--checkpoint-dir` and `--log-dir` set output paths.

Each run logs `console.log` (Verda launcher), `run.json` (effective configuration,
precision, hardware, resume source), and `metrics.jsonl` (one row per completed
cycle). Metrics include total/policy/Q losses, W/D/L, win rate, draw-adjusted
score rate, every historical matchup, position counts, timings and UTC/elapsed
time. No trajectories are saved. Opponent history rebuilds after resume.

After training finishes, run this **on your local machine** before deleting the
box. Replace `BOX_IP` and the SSH user/path if startup did not run as root:

```sh
mkdir -p verda-results
rsync -avP --include='/checkpoints/' --include='/checkpoints/***' \
  --include='/logs/' --include='/logs/***' --exclude='*' \
  root@BOX_IP:/root/projects/alpha-lines/ ./verda-results/
```

This downloads checkpoints and logs together and can be rerun to resume a
transfer. For a custom SSH port add `-e 'ssh -p PORT'` to rsync.
