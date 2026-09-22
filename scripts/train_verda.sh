#!/usr/bin/env bash
# Set up dependencies, then train in the current terminal.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
./scripts/setup_gpu.sh
# Fixed-width cycle names sort numerically. Include downloaded/local runs too.
checkpoint=$(find "$PWD/checkpoints" -type f -name 'cycle-??????.pt' -printf '%f\t%p\n' | sort -k1,1 | tail -n 1 | cut -f2-)
if [[ -z "$checkpoint" ]]; then
    echo 'No KLENT checkpoint found' >&2
    exit 1
fi
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
log_dir="$PWD/logs/klent/$run_id"
checkpoint_dir="$PWD/checkpoints/klent/$run_id"
mkdir -p "$log_dir"
echo "Resuming $checkpoint"
echo "Logs: $log_dir"
./scripts/train_local.sh --resume "$checkpoint" \
    --checkpoint-dir "$checkpoint_dir" --log-dir "$log_dir" \
    2>&1 | tee "$log_dir/console.log"
