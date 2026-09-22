#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Beam syncs the working directory. Do not upload checkpoints or the repository.
probe_dir="$(mktemp -d -t alphalines-beam-probe.XXXXXXXX)"
trap 'rm -rf -- "$probe_dir"' EXIT
cp "$script_dir/beam_h100_probe.py" "$probe_dir/probe.py"
cd "$probe_dir"
uv run --no-project --script probe.py
