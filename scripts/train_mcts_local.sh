#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
cargo build --release --locked --manifest-path rust/Cargo.toml
exec uv run --no-sync python -u -m mcts.train "$@"
