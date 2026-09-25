#!/usr/bin/env bash
# Download completed checkpoints and logs into this local checkout.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
    echo "Usage: $0 --ip BOX_IP"
}

box_ip=''
while (($#)); do
    case "$1" in
        --ip)
            if (($# < 2)) || [[ -z "$2" || "$2" == -* ]]; then
                usage >&2
                exit 1
            fi
            box_ip="$2"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
if [[ -z "$box_ip" || ! "$box_ip" =~ ^[a-zA-Z0-9.:]+$ ]]; then
    echo 'Provide the Verda box IP with --ip BOX_IP.' >&2
    exit 1
fi
# Brackets disambiguate IPv6 addresses in rsync's host:path syntax.
if [[ "$box_ip" == *:* ]]; then
    box_ip="[$box_ip]"
fi

exec rsync -avP \
    --exclude='*.pt.tmp' \
    --exclude='/checkpoints/klent/bf16/***' \
    --include='/checkpoints/' --include='/checkpoints/klent/***' --include='/checkpoints/mcts/***' \
    --include='/logs/' --include='/logs/***' --exclude='*' \
    "root@$box_ip:/root/projects/alpha-lines/" ./
