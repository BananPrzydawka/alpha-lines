"""Run one full MCTS training cycle on Modal: modal run -m mcts.modal_train."""

from pathlib import Path

import modal

from config import resources
from modal_image import base_image, project_dir, with_project_files, volume


app = modal.App('alphalines-mcts-pilot')

# Mount sources and the local KLENT checkpoint at run time. Build Rust in the
# container so this entrypoint uses the same native code as local training.
training_image = with_project_files(
    base_image.apt_install('build-essential')
    .run_commands('curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh',
                  'sh /tmp/rustup.sh -y --profile minimal')
).add_local_dir(
    project_dir / 'rust', remote_path='/root/rust', ignore=['target/']
).add_local_file(
    project_dir / 'checkpoints' / 'resume.pt', remote_path='/root/checkpoints/resume.pt'
)


@app.function(image=training_image, gpu=resources['gpu'], timeout=max(resources['timeout'], 7200),
              volumes={'/checkpoints': volume})
def train():
    import subprocess
    from uuid import uuid4

    from config import settings
    from mcts.train import run

    subprocess.run(['/root/.cargo/bin/cargo', 'build', '--release', '--locked',
                    '--manifest-path', '/root/rust/Cargo.toml'], check=True)

    run_id = uuid4().hex
    directory = Path('/checkpoints/mcts') / run_id
    print(f'MCTS pilot output: {directory}', flush=True)
    summaries = run('/root/rust/target/release/libalpha_lines_game.so',
                    options=settings['mcts'], cycles=1,
                    resume='/root/checkpoints/resume.pt',
                    checkpoint_dir=directory, log_dir=directory / 'logs')
    volume.commit()
    print(f'MCTS pilot committed: {directory}', flush=True)
    return {'directory': str(directory), 'cycle': summaries[-1]['cycle']}


@app.local_entrypoint()
def main():
    train.remote()
