#!/usr/bin/env bash
# Set up dependencies, then train in the current terminal.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
train_args=()
run_name=''
while (($#)); do
    case "$1" in
        --resume|--reference|--oponent|--opponent)
            if (($# < 2)) || [[ ! -f "$2" ]]; then
                echo "Expected an existing checkpoint after $1" >&2
                exit 1
            fi
            train_args+=("$1" "$2")
            shift 2 ;;
        --name)
            if (($# < 2)) || [[ ! "$2" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$ ]]; then
                echo 'Expected a safe run folder name after --name' >&2
                exit 1
            fi
            run_name="$2"
            shift 2 ;;
        --fresh) shift ;;
        *) echo 'Usage: ./scripts/train_verda.sh [--resume MODEL_PATH] [--reference MODEL_PATH ...] [--oponent MODEL_PATH] [--name RUN_NAME]' >&2; exit 1 ;;
    esac
done
./scripts/setup_gpu.sh
run_id="${run_name:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
log_dir="$PWD/logs/klent/$run_id"
checkpoint_dir="$PWD/checkpoints/klent/$run_id"
if [[ -e "$log_dir" || -e "$checkpoint_dir" ]]; then
    echo "Run folder already exists: $run_id" >&2
    exit 1
fi
mkdir -p "$log_dir"
echo "Logs: $log_dir"
./scripts/train_local.sh "${train_args[@]}" \
    --checkpoint-dir "$checkpoint_dir" --log-dir "$log_dir" \
    2>&1 | tee "$log_dir/console.log"
