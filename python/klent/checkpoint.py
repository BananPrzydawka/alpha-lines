"""Atomic, cycle-boundary training snapshots."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from config import settings


def _cpu_copy(value, tensors):
    if isinstance(value, torch.Tensor):
        key = id(value)
        if key not in tensors:
            tensors[key] = value.detach().to('cpu', copy=True)
        return tensors[key]
    if isinstance(value, dict):
        return {key: _cpu_copy(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item, tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item, tensors) for item in value)
    return value


def capture(model, optimizer, options, summary, anchors=(), training_opponent=None):
    """Freeze all checkpoint data before the next optimizer step."""
    if not anchors or anchors[0][0] > summary['cycle']:
        raise ValueError('KLENT checkpoints require an initial anchor')
    tensors = {}
    return dict(format_version=2, model=_cpu_copy(model.state_dict(), tensors),
        optimizer=_cpu_copy(optimizer.state_dict(), tensors),
        options={key: value for key, value in options.items() if key != 'opponent_checkpoint'},
        model_config=dict(settings[options['model']+'_model']), summary=dict(summary),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        # Native arena RNG is not serialized; this is not an exact replay snapshot.
        exact_resume=False)


def write(directory, snapshot):
    """Serialize one frozen checkpoint and atomically publish its file."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cycle-{snapshot['summary']['cycle']:06d}.pt"
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {path}')
    temporary = path.with_suffix('.pt.tmp')
    torch.save(snapshot, temporary)
    temporary.replace(path)
    return path


def save(directory, model, optimizer, options, summary, anchors=(), training_opponent=None):
    return write(directory, capture(model, optimizer, options, summary, anchors, training_opponent))


class AsyncWriter:
    """Keep at most one pending write in this training container."""
    def __init__(self, directory, commit):
        self.directory = directory
        self.commit = commit
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='klent-checkpoint')
        self.pending = None

    def save(self, model, optimizer, options, summary, anchors, training_opponent=None):
        if self.pending is not None:
            self.pending.result()
        snapshot = capture(model, optimizer, options, summary, anchors, training_opponent)
        self.pending = self.pool.submit(self._write, snapshot)

    def _write(self, snapshot):
        path = write(self.directory, snapshot)
        self.commit()
        print(f'Checkpoint saved: {path}', flush=True)

    def close(self):
        try:
            if self.pending is not None:
                self.pending.result()
        finally:
            self.pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
