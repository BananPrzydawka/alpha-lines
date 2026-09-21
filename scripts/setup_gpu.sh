#!/usr/bin/env bash
# Ubuntu GPU image with a working NVIDIA driver already installed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $EUID -eq 0 ]]; then
    apt-get update
    apt-get install -y git curl build-essential tmux rsync ca-certificates
else
    sudo apt-get update
    sudo apt-get install -y git curl build-essential tmux rsync ca-certificates
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh -o /tmp/alpha-lines-uv-install.sh
    sh /tmp/alpha-lines-uv-install.sh
fi
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/alpha-lines-rustup.sh
    sh /tmp/alpha-lines-rustup.sh -y --profile minimal
fi
uv python install 3.13
uv sync --locked --python 3.13
cargo build --release --locked --manifest-path rust/Cargo.toml
nvidia-smi
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check NVIDIA driver"; assert torch.cuda.is_bf16_supported(), "GPU must support BF16"; x=torch.ones(1,device="cuda"); print("GPU:",torch.cuda.get_device_name(),"torch:",torch.__version__,"CUDA:",torch.version.cuda)'
