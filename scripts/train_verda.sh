#!/usr/bin/env bash
# Start once from the instance startup script; attach later with tmux.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if tmux has-session -t klent 2>/dev/null; then
    echo 'The klent tmux session already exists; attach with: tmux attach -t klent'
    exit 0
fi
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
# Keep an interactive shell open after completion/failure so output is inspectable.
printf -v command 'set -o pipefail; %q --resume %q --checkpoint-dir %q --log-dir %q 2>&1 | tee %q; result=${PIPESTATUS[0]}; echo "Training exit status: $result"; exec bash' \
    "$PWD/scripts/train_local.sh" "$checkpoint" "$checkpoint_dir" "$log_dir" "$log_dir/console.log"
initial_window=$(tmux new-session -d -P -F '#{window_id}' -s klent -c "$PWD")
tmux set-option -t klent history-limit 100000
tmux set-option -t klent mouse on
# history-limit takes effect when a pane is created.
tmux new-window -t klent: -c "$PWD"
tmux kill-window -t "$initial_window"
tmux send-keys -t klent -l "bash -c $(printf '%q' "$command")"
tmux send-keys -t klent Enter
echo "Resuming $checkpoint"
echo "Logs: $log_dir"
echo 'Attach: tmux attach -t klent (detach: Ctrl-b then d; scroll: Ctrl-b then [)'
