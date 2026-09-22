# Alpha Lines

Rust implements the game engine and MCTS. Python provides the model, native
inference export, and a score-learning demo on Modal.

## KLENT training on a rented GPU

For Ubuntu GPU boxes over SSH, run `./scripts/train_verda.sh` after cloning.
It completes setup and resumes the latest checkpoint in the current terminal.
Set additional cycles in `config.json` (`klent.cycles`). The seed checkpoint
`checkpoints/cycle-000166.pt` ships in Git; new checkpoints and metrics stay local.
See [KLENT instructions](python/klent/README.md) for fresh runs, checkpointing,
historical evaluation and benchmarking.

## Setup on Fedora

```bash
sudo dnf install uv rust cargo gcc gcc-c++
uv python install 3.13
uv sync --locked --managed-python
uv run modal token new
```

## Configuration

Edit `config.json` at the repository root. It contains the board dimensions,
model architecture, MCTS defaults, and Modal resources.
`python/config.py` only loads this file. Rust's build script reads the same file
and generates constants; Cargo rebuilds them whenever the file changes. Run via
Cargo after editing configuration, rather than running an old compiled binary.

The current engine and model encoder support only a 10×16 board. Both config
readers reject other dimensions. MCTS node capacity defaults to
`g * s * node_capacity_factor`.

## Model inference

`python/export_mcts.py` is currently disabled: the native MCTS contract needs
updating for spatial action values. The commands below describe the previous
export workflow. The optional Rust `compiled-model` feature loads packages
through the native inference bridge.

```bash
uv run python python/export_mcts.py --batch 2048 --output model.pt2
```

Export on the GPU used for inference. `--batch` counts positions; the model
processes two player perspectives per position. `--seed` defaults to 1 and
`--device` to CUDA. CPU export is available for smoke checks.

`python/modal_image.py` defines the shared CUDA image and output volume. Local
direct dependencies are Modal, PyTorch, torchinfo, and NumPy.

Print layer shapes, parameter counts, multiply-adds, and estimated parameter and
forward/backward memory using the current `config.json`:

```bash
uv run python python/model_summary.py resnet
uv run python python/model_summary.py maia --batch 32
uv run python python/model_summary.py katago --depth 2
```

Summaries run on CPU with BF16 weights and inputs. Memory figures are estimates, not measured peak
usage; functional operations such as attention are not fully accounted for.

The core Rust engine can be built without local PyTorch or CUDA:

```bash
cargo build --manifest-path rust/Cargo.toml
```

The build-time JSON parser is a Cargo dependency, not a runtime engine dependency.

## Current-board score learning demo

```bash
PYTHONPATH=python uv run modal run -m scores.modal_scores
# Short GPU smoke run:
PYTHONPATH=python uv run modal run -m scores.modal_scores --batch 32 --steps 10 --eval-every 5 --validation-steps 64
```

Defaults live in `config.json` under `score_training`; Modal GPU, CPU, memory and
timeout use the existing `modal` section. `batch` is the number of boards, and
`steps` is an optional upper limit on optimizer updates (default 100,000).
The default run uses 256 boards and a 300-second budget, adjustable with
`--seconds`. The timer starts before initial evaluation and includes compilation,
validation, and reporting. Artifacts are saved once after training finishes. No new update starts after the deadline; an
in-flight update and final evaluation/save may finish afterward. Provisioning,
Rust build, data setup and local downloads are outside this budget.
The demo trains from fresh random weights.

`rust/src/score_batch.rs` owns a vector of games, one RNG and one scratch buffer.
Each call independently samples both players' legal moves uniformly from the
same pre-move board, applies the simultaneous move, and returns the resulting
cells and exact current scores. The engine's opening half-board restriction and
collisions apply normally. Terminal boards are trained on once; their slots
restart on the next call. Initial empty boards are omitted. There is no arena,
search, replay buffer, data-loader worker or game-generation thread. A small C
ABI writes directly into reusable PyTorch CPU tensors through `ctypes`.

The score-prediction harness now supports only `models.maia.MaiaNet()`.
It takes five planes (unplayable, empty, removed, player 0 marks, player 1 marks)
and returns `[batch, 2, 81]` score logits. Scores are targets, never inputs to Maia.
ResNet and KataGo use the action-model interface described below.

Training creates two examples per position: original and player-swapped.
Player swapping exchanges the two mark planes and reverses the score targets;
boards are never rotated. Each output predicts the score of the player in its
mark plane. With 256 game slots, the training model batch is 512 examples,
matching the pre-rotation experiment, with one loss averaged over all 1,024 score
predictions and one optimizer update. The logged `boards` counts original
positions; saved metrics also record `examples`, which is twice that count.
Validation keeps the original player order and original positions for comparison
with earlier runs; it does not average predictions from two perspectives.

Every step performs forward, cross-entropy averaged over both players and all
boards, backward, and one AdamW update. CUDA uses BF16 autocast with FP32 weights.
On CUDA the model uses `torch.compile(mode="default", fullgraph=True,
dynamic=False)`, without max autotune. The first validation and training calls
include compilation time; CPU smoke checks stay eager. Checkpoints retain the
original model's parameter names.
Validation uses fixed random trajectories from a separate seed, without gradient
updates. A separate batch of 256 games advances 64 times, retaining every
post-move board (16,384 positions). Finished slots restart independently; the
set is not explicitly balanced by stage. `--validation-batch` and
`--validation-steps` control these dimensions independently of training.
Validation positions receive no augmentation. It reports loss, each player's exact-score accuracy and argmax mean
absolute error, and an always-zero baseline. Validation is recorded before
training and every `eval_every` updates, including the last update. The default
64 validation steps cover full games and restarts; very short validation runs
mostly measure low-scoring opening positions. Fixed validation allows comparison
over time, but repeated tuning against it would require a fresh test set.

Modal saves `metrics.csv`, `results.json`, and `checkpoint.pt` to a unique
`score-demo/<run-id>` directory in the `alphalines` volume and commits once at the end of training. The checkpoint includes model and optimizer states, config,
step and input/output conventions; it does not include game/RNG state for exact
resumption. The entrypoint downloads the final `checkpoint.pt` and `metrics.csv` and saves
returned metrics in `results.json`, all under `logs/score-symmetry/<run-id>/`.
`--output` changes the parent directory; every run gets a unique subdirectory
so repeated invocations preserve previous results.
Console reports show the learning curve as it
runs. Improvement on held-out games must be measured; the smoke tests only check
correctness and that the model can fit a small fixed batch.

Local CPU checks (no Modal GPU required):

```bash
cargo test --manifest-path rust/Cargo.toml score_batch --lib
cargo build --manifest-path rust/Cargo.toml --release --lib
PYTHONPATH=python uv run python -m unittest discover -s python -p test_scores.py
```

## Compare validation loss over time

```bash
uv run --script python/scores/plot.py
# Or choose a run collection and output format:
uv run --script python/scores/plot.py logs/score-symmetry --output /tmp/comparison.svg
```

The standalone script installs its plotting dependencies through uv, recursively
finds runs, and overlays their validation losses against elapsed seconds. It
prefers `results.json`, falling back to `metrics.csv` when JSON is absent, so a
run with both files is plotted once. Run labels are relative directory names.
It saves `logs/score-symmetry/validation-loss.png` by default; an empty collection
produces a graph marked "No saved runs yet". Rerun after downloads to update it.
Use `--log-y` to inspect losses when large early spikes compress the later curves.

### Maia-style alternative for score experiments

Run `PYTHONPATH=python uv run modal run -m scores.modal_scores --model maia` to select the
model in `python/models/maia.py`; `maia` is the default and only supported score model.
It uses the configured training batch, player-perspective duplication, validation
positions, optimizer, and time budget. The chosen architecture is saved in run
options and checkpoints. Training hyperparameters remain your configured values,
not the paper's human-imitation learning-rate schedule.

`config.json` → `maia_model` controls the Maia-3 small-model adaptation:
configurable encoder depth, width 192, 6 attention heads of dimension 32, MLP expansion 2,
and pooled GAB with hidden/template dimensions 64. It uses only the 80 playable
squares, in fixed board order, without history or rating embeddings. Every layer
mixes a shared bank of learned 80×80 attention-bias templates using coefficients
generated from its mean-pooled board representation. Pre-LayerNorm residuals,
GELU MLPs, and no dropout are implementation choices where the supplied paper
leaves details unspecified. There is no additional absolute or relative encoding.

Score mode returns `(batch, 2, 81)` score logits. Its score head applies
LayerNorm and a shared projection to 32 features per square, then flattens the
80 square representations into a 256-unit GELU MLP. Configure these widths with
`maia_model.score_head.square_features` and `hidden_dim`. Policy/value head definitions and their forward operations are commented out
in `models/maia.py`; the active model has only a score head and five input planes.
Maia retains its existing optional `apply_softmax` argument.

Run `PYTHONPATH=python .venv/bin/python -m unittest discover -s python/scores -p test_maia.py`
for the alternative model's CPU checks, after building the Rust library.

### Python layout

- `python/models/resnet.py`: score-conditioned ResNet with spatial policy and action-value heads.
- `python/models/maia.py`: Maia score model with unused heads commented out.
- `python/scores/`: data generation bridge, training, Modal entrypoint, plotting, and tests.
- `python/config.py`, `modal_image.py`, and `export_mcts.py`: shared setup and native export.

Run all score checks with
`PYTHONPATH=python .venv/bin/python -m unittest discover -s python/scores -p 'test_*.py'`.

Score progress lines show total milliseconds in each phase since the previous
report (since startup at step zero), without dividing by update count. Saved results also retain cumulative
seconds. Phase times: `cpu` covers
training data generation, transfers, encoding, and perspective duplication;
`model` covers training forward/loss/backward and optimizer updates; `log`
covers validation and reporting. Final artifact writes and the volume commit are
outside the reported phase timings. Compilation
is charged to the phase that triggers it. CUDA is synchronized at phase boundaries
for accurate attribution, which can add overhead. Each snapshot includes its
validation time; its own terminal output appears in the next snapshot. Initial setup remains outside the training budget.
MAE metrics remain in saved results but are omitted from the terminal line.

### ResNet and KataGo action models

Each network has fully separate settings: `resnet_model`, `katago_model`, and
`maia_model`. The convolutional networks independently configure `filters`,
`blocks`, `se_hidden`, `score_embed_hidden`, `policy_filters`,
`action_value_filters`, and `group_norm` (`groups`, `eps`, `affine`).
GroupNorm uses `gcd(groups, channels)` groups.

```python
policy_logits, action_values = net(board, scores)
```

`board` has shape `(batch, 5, 10, 16)`, and `scores` has shape `(batch, 2)`
with raw scores from 0 to 80 ordered current player, opponent. The board's
player mark planes must use the same order. An unbatched board and score pair
are also accepted and produce a batch of one.

Scores are divided by 80 and passed through Linear → SiLU → Linear.
The resulting channel embedding is broadcast-added after the initial convolution
and GroupNorm, before SiLU. Both heads use 1×1 convolution → GroupNorm → SiLU
→ 1×1 convolution and return `(batch, 10, 16)`. Policy logits have no softmax;
action values have no output activation. Legal-action masking belongs to the caller.
There are no other heads in these two models. Old score-head checkpoints are incompatible.

KataGo's outer blocks project to half width, apply two two-convolution inner
residual blocks, project back, apply SE, and add the outer skip. This is a
KataGo-inspired adaptation using GroupNorm and SiLU. Its settings are independent
of ResNet, including normalization and head widths.

The score-training harness is reserved for Maia. Native MCTS export remains
inactive because the Rust interface expects scalar state values and needs updating
for spatial action values. Summary and inference benchmark tools support all three
networks with their respective input and output formats.

Run action-model checks with
`PYTHONPATH=python .venv/bin/python -m unittest discover -s python/models -p 'test_*.py'`.
