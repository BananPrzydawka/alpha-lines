"""Run KLENT on Modal: modal run -m klent.modal_train."""
import os
import re
import threading
from pathlib import Path
from uuid import uuid4

import modal
from config import resources
from modal_image import base_image, project_dir, with_project_files, volume

app = modal.App('alphalines-klent')
compile_cache = modal.Volume.from_name('alphalines-klent-compile-cache', create_if_missing=True)


def validate_run_name(name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}', name):
        raise ValueError('--name must be 1–100 characters: letters, digits, dots, _ or -')


def remote_run_files(run_name, source=volume):
    try:
        return source.listdir(f'klent/{run_name}')
    except (FileNotFoundError, modal.exception.NotFoundError):
        return []


def pull_run_files(run_name, source=volume, root=project_dir):
    """Download committed files to the existing checkpoint/log layout."""
    remote_dir = f'klent/{run_name}'
    checkpoint_dir = Path(root)/'checkpoints'/'klent'/run_name
    log_dir = Path(root)/'logs'/'klent'/run_name
    pulled = []
    for entry in remote_run_files(run_name, source):
        if entry.type.name != 'FILE':
            continue
        name = Path(entry.path).name
        if name.endswith('.tmp'):
            continue
        local_dir = checkpoint_dir if name.endswith('.pt') else log_dir
        local_dir.mkdir(parents=True, exist_ok=True)
        target = local_dir/name
        if target.exists() and target.stat().st_size == entry.size:
            continue
        remote_file = entry.path if entry.path.startswith(remote_dir + '/') else f'{remote_dir}/{name}'
        temporary = target.with_name(target.name + '.download')
        try:
            size = 0
            with temporary.open('wb') as stream:
                for chunk in source.read_file(remote_file):
                    stream.write(chunk)
                    size += len(chunk)
            if size != entry.size:
                raise IOError(f'Incomplete Modal download: {remote_file} ({size}/{entry.size} bytes)')
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        pulled.append(target)
    return pulled


def pull_while_training(run_name, stop):
    warning = None
    while not stop.is_set():
        try:
            for path in pull_run_files(run_name):
                print(f'Pulled: {path}', flush=True)
            warning = None
        except Exception as error:
            message = str(error)
            if message != warning:
                print(f'Modal pull retrying: {message}', flush=True)
                warning = message
        stop.wait(5)


# Runtime-mounted sources keep iteration cheap. The toolchain is an image layer;
# the small native library is compiled when the function starts.
training_image = (
    with_project_files(
        base_image.apt_install('build-essential')
        .run_commands('curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh',
                      'sh /tmp/rustup.sh -y --profile minimal')
        .env({'TORCHINDUCTOR_CACHE_DIR': '/compile-cache/inductor',
              'TRITON_CACHE_DIR': '/compile-cache/triton',
              'CUDA_CACHE_PATH': '/compile-cache/cuda'})
    ).add_local_dir(project_dir/'rust',remote_path='/root/rust',ignore=['target/'])
)


@app.function(image=training_image,gpu=resources['gpu'],timeout=resources['timeout'],
              volumes={'/checkpoints': volume, '/compile-cache': compile_cache},
              max_containers=1, buffer_containers=0, scaledown_window=2)
def train(smoke: bool = False, resume: str = '', references: tuple[str, ...] = (),
          opponent: str = '', run_name: str = ''):
    import subprocess
    from config import settings
    from klent.train import run
    subprocess.run(['/root/.cargo/bin/cargo','build','--release','--locked',
                    '--manifest-path','/root/rust/Cargo.toml'],check=True)
    options = dict(settings['klent'])
    if smoke:
        options.update(n=8,m=1280,train_minibatch=32,test_games=8,cycles=1)
    from klent.checkpoint import AsyncWriter
    run_name = run_name or uuid4().hex
    validate_run_name(run_name)
    directory = '/checkpoints/klent/' + run_name
    if Path(directory).exists():
        raise FileExistsError(f'Run folder already exists on Modal: {directory}')
    try:
        with AsyncWriter(directory, volume.commit) as writer:
            return run('/root/rust/target/release/libalpha_lines_game.so',options,
                       checkpoint=writer.save,log_dir=directory,
                       resume=resume or None,
                       references=references if references else None,
                       opponent=opponent or None,
                       max_seconds=resources['timeout']-60)
    finally:
        if options.get('compile_model', True):
            compile_cache.commit()


@app.local_entrypoint()
def main(*args: str):
    import argparse
    parser = argparse.ArgumentParser(description='Run KLENT training on Modal')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', default='', metavar='MODEL_PATH')
    parser.add_argument('--reference', action='append', default=[], metavar='MODEL_PATH',
                        help='Repeat to test against multiple checkpoint models')
    parser.add_argument('--oponent', '--opponent', dest='opponent', default='',
                        metavar='MODEL_PATH', help='Train against this exact checkpoint')
    parser.add_argument('--name', default='', metavar='RUN_NAME',
                        help='Name the run folder under checkpoints/klent and logs/klent')
    parsed = parser.parse_args(args)
    run_name = parsed.name or uuid4().hex
    try:
        validate_run_name(run_name)
    except ValueError as error:
        parser.error(str(error))
    checkpoint_dir = project_dir/'checkpoints'/'klent'/run_name
    log_dir = project_dir/'logs'/'klent'/run_name
    if checkpoint_dir.exists() or log_dir.exists():
        parser.error(f'Local run folder already exists: {run_name}')
    if remote_run_files(run_name):
        parser.error(f'Modal run folder already exists: {run_name}')
    paths = ([parsed.resume] if parsed.resume else []) + parsed.reference + (
        [parsed.opponent] if parsed.opponent else [])
    staged = {}
    local_paths = {path: Path(path) for path in paths
                   if not path.startswith('/checkpoints/')}
    for path, local in local_paths.items():
        if not local.is_file():
            parser.error(f'Checkpoint does not exist locally or on /checkpoints: {path}')
    if local_paths:
        prefix = f'imports/{uuid4().hex}'
        with volume.batch_upload() as upload:
            for index, (path, local) in enumerate(local_paths.items()):
                remote = f'{prefix}/{index}-{local.name}'
                upload.put_file(str(local), remote)
                staged[path] = f'/checkpoints/{remote}'
    stop = threading.Event()
    puller = threading.Thread(target=pull_while_training, args=(run_name, stop),
                              name='klent-modal-pull', daemon=True)
    print(f'Local checkpoints: {checkpoint_dir}', flush=True)
    print(f'Local logs: {log_dir}', flush=True)
    puller.start()
    training_failed = False
    try:
        train.remote(parsed.smoke, staged.get(parsed.resume, parsed.resume),
                     tuple(staged.get(path,path) for path in parsed.reference),
                     staged.get(parsed.opponent, parsed.opponent), run_name)
    except BaseException:
        training_failed = True
        raise
    finally:
        stop.set()
        puller.join()
        try:
            for path in pull_run_files(run_name):
                print(f'Pulled: {path}', flush=True)
        except Exception as error:
            if not training_failed:
                raise
            print(f'Final Modal pull failed: {error}', flush=True)
