# Alpha Lines

The Rust engine runs parallel MCTS arenas against a random-weight PyTorch model
on a Modal GPU. Python exports the model and launches the benchmark.

## Setup on Fedora

```bash
sudo dnf install uv rust cargo gcc gcc-c++
uv python install 3.13
uv sync --locked --managed-python
uv run modal token new
```

## Configuration

Edit `config.json` at the repository root. It contains the board dimensions,
model architecture, MCTS defaults, benchmark settings, and Modal resources.
`python/config.py` only loads this file. Rust's build script reads the same file
and generates constants; Cargo rebuilds them whenever the file changes. Run via
Cargo after editing configuration, rather than running an old compiled binary.

The current engine and model encoder support only a 10×16 board. Both config
readers reject other dimensions. MCTS node capacity defaults to
`g * s * node_capacity_factor`; the native benchmark can override it with
`--node-capacity`.

CLI arguments override shared defaults for that run. `searches` is the Modal
sweep of arena counts; the native executable runs one count using `--workers`
(default 1). Native `--steps` selects a fixed-step run instead of the configured
duration; if both `--steps` and `--seconds` are supplied, the last one wins.

## Benchmark

Show options without launching GPU work:

```bash
uv run modal run python/modal_mcts.py --help
```

Run the configured sweep:

```bash
uv run modal run python/modal_mcts.py
```

`python/modal_image.py` defines the shared image and output volume. It mounts the
same root config used locally. The image explicitly installs only CUDA-enabled
PyTorch and curl (for remote Rust setup), on a CUDA development base that supplies
the native compilation tools and headers. Modal supplies its own client.
Local direct dependencies are only Modal and PyTorch.
The remote runner builds Rust with `compiled-model` and caches exported models
by source, config, PyTorch version, GPU, and batch size. Results are written to
the configured output directory as JSON and CSV.

The core Rust engine can be built without local PyTorch or CUDA:

```bash
cargo build --manifest-path rust/Cargo.toml
```

The build-time JSON parser is a Cargo dependency, not a runtime engine dependency.

## Current-board score learning demo

```bash
uv run modal run python/modal_scores.py
# Short GPU smoke run:
uv run modal run python/modal_scores.py --batch 32 --steps 10 --eval-every 5 --validation-steps 64
```

Defaults live in `config.json` under `score_training`; Modal GPU, CPU, memory and
timeout use the existing `modal` section. `batch` is the number of boards, and
`steps` is an optional upper limit on optimizer updates (default 100,000).
The default run uses 256 boards and a 600-second budget, adjustable with
`--seconds`. The timer starts before initial evaluation and includes compilation,
validation, and checkpoint saves. No new update starts after the deadline; an
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

`alpha_lines_net(score_prediction=True)` uses the configured residual tower,
replaces the input convolution with five channels, and attaches a score head
producing `[batch, 2, 81]` raw logits. The five planes are unplayable, empty,
removed, player 0 marks, and player 1 marks. Scores and internal reachability
levels are never inputs. Outputs always mean player 0 and player 1, with classes
0 through 80 inclusive. This opt-in mode leaves the default MCTS model interface
available. No player-perspective duplication is needed.

Every step performs forward, cross-entropy averaged over both players and all
boards, backward, and one AdamW update. CUDA uses BF16 autocast with FP32 weights.
On CUDA the model uses `torch.compile(mode="default", fullgraph=True,
dynamic=False)`, without max autotune. The first validation and training calls
include compilation time; CPU smoke checks stay eager. Checkpoints retain the
original model's parameter names.
Validation uses fixed random trajectories from a separate seed, without gradient
updates. It reports loss, each player's exact-score accuracy and argmax mean
absolute error, and an always-zero baseline. Validation is recorded before
training and every `eval_every` updates, including the last update. The default
64 validation steps cover full games and restarts; very short validation runs
mostly measure low-scoring opening positions. Fixed validation allows comparison
over time, but repeated tuning against it would require a fresh test set.

Modal saves `metrics.csv`, `results.json`, and `checkpoint.pt` to a unique
`score-demo/<run-id>` directory in the `alphalines` volume and commits at each
validation point. The checkpoint includes model and optimizer states, config,
step and input/output conventions; it does not include game/RNG state for exact
resumption. The entrypoint downloads the final `checkpoint.pt` and `metrics.csv` and saves
returned metrics in `results.json`, all under `logs/score-demo/` (or `--output`).
Use a different output directory for each experiment to retain local models.
Console reports show the learning curve as it
runs. Improvement on held-out games must be measured; the smoke tests only check
correctness and that the model can fit a small fixed batch.

Local CPU checks (no Modal GPU required):

```bash
cargo test --manifest-path rust/Cargo.toml score_batch --lib
cargo build --manifest-path rust/Cargo.toml --release --lib
PYTHONPATH=python uv run python -m unittest discover -s python -p test_scores.py
```
