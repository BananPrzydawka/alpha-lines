"""Atomic, cycle-boundary training snapshots."""
from pathlib import Path
import torch
from config import settings


def anchor_path(directory, cycle):
    return Path(directory)/'anchors'/f'anchor-{cycle:06d}.pt'


def save(directory, model, optimizer, options, summary, anchors=()):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cycle-{summary['cycle']:06d}.pt"
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite checkpoint: {path}')
    for cycle, weights in anchors:
        anchor = anchor_path(directory, cycle)
        if not anchor.exists():
            anchor.parent.mkdir(parents=True, exist_ok=True)
            temporary_anchor = anchor.with_suffix('.pt.tmp')
            torch.save({k: v.detach().cpu() for k,v in weights.items()}, temporary_anchor)
            temporary_anchor.replace(anchor)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(dict(format_version=1, model=model.state_dict(),
        optimizer=optimizer.state_dict(), options=dict(options),
        model_config=dict(settings[options['model']+'_model']), summary=dict(summary),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        # Native arena RNG is not serialized; this is not an exact replay snapshot.
        exact_resume=False), temporary)
    temporary.replace(path)
    return path
