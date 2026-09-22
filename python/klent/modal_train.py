"""Run KLENT on Modal: modal run -m klent.modal_train."""
import modal
from config import resources
from modal_image import base_image, project_dir, with_project_files, volume

app = modal.App('alphalines-klent')
# Runtime-mounted sources keep iteration cheap. The toolchain is an image layer;
# the small native library is compiled when the function starts.
training_image = with_project_files(
    base_image.apt_install('build-essential')
    .run_commands('curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh',
                  'sh /tmp/rustup.sh -y --profile minimal')
).add_local_dir(project_dir/'rust',remote_path='/root/rust',ignore=['target/'])


@app.function(image=training_image,gpu=resources['gpu'],timeout=resources['timeout'],
              volumes={'/checkpoints': volume})
def train(smoke: bool = False):
    import subprocess
    from config import settings
    from klent.train import run
    subprocess.run(['/root/.cargo/bin/cargo','build','--release','--locked',
                    '--manifest-path','/root/rust/Cargo.toml'],check=True)
    options = dict(settings['klent'])
    if smoke:
        options.update(n=8,m=1280,train_minibatch=32,test_games=8)
    from uuid import uuid4
    from klent.checkpoint import save
    directory = '/checkpoints/klent/' + uuid4().hex
    def checkpoint(model, optimizer, options, summary):
        path = save(directory, model, optimizer, options, summary)
        volume.commit()
        print(f'Checkpoint saved: {path}', flush=True)
    return run('/root/rust/target/release/libalpha_lines_game.so',options,
               checkpoint=checkpoint,log_dir=directory)


@app.local_entrypoint()
def main(smoke: bool = False):
    train.remote(smoke)
