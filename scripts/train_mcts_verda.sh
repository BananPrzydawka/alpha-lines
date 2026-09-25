#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
checkpoint="$PWD/checkpoints/resume.pt"
if [[ ! -f "$checkpoint" ]]; then
    echo 'Place the chosen FP32 KLENT checkpoint at checkpoints/resume.pt.' >&2
    exit 1
fi
./scripts/setup_gpu.sh
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
log_dir="$PWD/logs/mcts/$run_id"
checkpoint_dir="$PWD/checkpoints/mcts/$run_id"
mkdir -p "$log_dir"
echo "Resuming $checkpoint"
echo "Logs: $log_dir"
./scripts/train_mcts_local.sh --resume "$checkpoint" \
    --checkpoint-dir "$checkpoint_dir" --log-dir "$log_dir" \
    2>&1 | tee "$log_dir/console.log"
