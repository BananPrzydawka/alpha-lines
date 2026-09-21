"""Shared Modal image and output volume for GPU workloads."""
from pathlib import Path

import modal

project_dir = Path(__file__).resolve().parent.parent
volume = modal.Volume.from_name("alphalines", create_if_missing=True)
base_image = (
    # The native loader and AOTInductor export compile against CUDA headers.
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.13")
    # rustup uses curl when provisioning the cached remote Rust toolchain.
    .apt_install("curl")
    .pip_install("torch==2.12.1", index_url="https://download.pytorch.org/whl/cu130")
    # Entrypoint sibling modules live in our explicit /root/python mount.
    .env({"PYTHONPATH": "/root/python"})
)


def with_project_files(base):
    """Attach runtime sources after all image build steps are complete."""
    return (
        base.add_local_dir(project_dir / "python", remote_path="/root/python", ignore=["__pycache__/"])
        .add_local_file(project_dir / "config.json", remote_path="/root/config.json")
    )


image = with_project_files(base_image)
