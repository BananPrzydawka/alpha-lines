#!/usr/bin/env bash
# Set up dependencies, then train in the current terminal.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fresh=false
if [[ "${1:-}" == '--fresh' ]]; then
    fresh=true
    shift
fi
if (($#)); then
    echo 'Usage: ./scripts/train_verda.sh [--fresh]' >&2
    exit 1
fi
# This input is chosen manually; run outputs never replace it.
checkpoint="$PWD/checkpoints/resume.pt"
if [[ "$fresh" == false && ! -f "$checkpoint" ]]; then
    echo 'Place your chosen FP32 checkpoint at checkpoints/resume.pt and commit it before deploying.' >&2
    exit 1
fi
if [[ ! -f "$PWD/checkpoints/349.pt" ]]; then
    echo 'Place the fixed evaluation opponent at checkpoints/349.pt before training.' >&2
    exit 1
fi
./scripts/setup_gpu.sh
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
log_dir="$PWD/logs/klent/$run_id"
checkpoint_dir="$PWD/checkpoints/klent/$run_id"
mkdir -p "$log_dir"
train_args=()
if [[ "$fresh" == true ]]; then
    echo 'Starting a fresh KLENT model'
else
    train_args+=(--resume "$checkpoint")
    echo "Resuming $checkpoint"
fi
echo "Logs: $log_dir"
./scripts/train_local.sh "${train_args[@]}" \
    --checkpoint-dir "$checkpoint_dir" --log-dir "$log_dir" \
    2>&1 | tee "$log_dir/console.log"
