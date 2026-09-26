#!/usr/bin/env bash
# Set up dependencies, then train in the current terminal.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
resume=false
case "${1:-}" in
    --resume) resume=true; shift ;;
    --fresh) shift ;; # Backward-compatible spelling; fresh is the default.
esac
if (($#)); then
    echo 'Usage: ./scripts/train_verda.sh [--resume|--fresh]' >&2
    exit 1
fi
# Resuming is explicit; fresh training needs only the fixed evaluation model.
checkpoint="$PWD/checkpoints/resume.pt"
if [[ "$resume" == true && ! -f "$checkpoint" ]]; then
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
if [[ "$resume" == true ]]; then
    train_args+=(--resume "$checkpoint")
    echo "Resuming $checkpoint"
else
    echo 'Starting a fresh KLENT model'
fi
echo "Logs: $log_dir"
./scripts/train_local.sh "${train_args[@]}" \
    --checkpoint-dir "$checkpoint_dir" --log-dir "$log_dir" \
    2>&1 | tee "$log_dir/console.log"
