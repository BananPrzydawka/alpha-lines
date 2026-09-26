"""Compare KLENT training gradients from independent self-play arenas."""
from contextlib import nullcontext
from pathlib import Path

import torch

from config import settings
from klent.native import Arena, Batch
from klent.train import load_core_weights, losses, recalculated_policy, validate
from models.factory import build_model


def _gradient_samples(library, model, options, device, seed, chunks, indices):
    rows = options['train_minibatch']
    positions_per_chunk = rows // 2
    needed = chunks * positions_per_chunk
    gradients = torch.empty((chunks, sum(p.numel() for p in model.parameters())))
    batch = Batch(rows, pinned=device.type == 'cuda')
    valid = torch.ones(rows, device=device)
    model.eval()
    with Arena(library, options, seed=seed, pinned=device.type == 'cuda') as arena:
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            while True:
                boards = arena.inputs().to(device, non_blocking=device.type == 'cuda')
                logits, values = model(boards.contiguous(memory_format=torch.channels_last))
                pi = logits.flatten(1)[:, indices].float().cpu().contiguous()
                q = values.flatten(1)[:, indices].float().cpu().contiguous()
                if arena.step(pi, q):
                    break
        count = arena.stats().positions
        if count < needed:
            raise ValueError(f'Collected {count} positions, need {needed}; increase klent.m')
        arena.shuffle()
        model.train()
        for chunk in range(chunks):
            size = arena.batch(chunk * positions_per_chunk, batch)
            if size != positions_per_chunk:
                raise RuntimeError('Incomplete gradient sample')
            boards, stored_target, actions, returns = (
                tensor.to(device, non_blocking=device.type == 'cuda') for tensor in batch.tensors
            )
            boards = boards.contiguous(memory_format=torch.channels_last)
            model.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16):
                logits, values = model(boards)
                target = (recalculated_policy(logits, values, boards, valid,
                          options['alpha'], options['beta'])
                          if options['policy_recalculation'] else stored_target)
                policy_loss, value_loss = losses(logits, values, target, actions, returns, valid)
                loss = options['policy_loss_weight'] * policy_loss + options['q_loss_weight'] * value_loss
            loss.backward()
            gradient = torch.cat([
                p.grad.detach().flatten() if p.grad is not None else torch.zeros_like(p).flatten()
                for p in model.parameters()
            ]).float().cpu()
            if not torch.isfinite(gradient).all():
                raise RuntimeError('Nonfinite gradient')
            gradients[chunk].copy_(gradient)
        model.zero_grad(set_to_none=True)
    return gradients, count


def compare(library, checkpoint, *, options=None, chunks=16, device='cuda'):
    """Measure gradient agreement at effective batches of 1, 2, 4, ... minibatches."""
    if type(chunks) is not int or chunks < 2 or chunks & (chunks - 1):
        raise ValueError('chunks must be a power of two of at least two')
    options = dict(settings['klent'] if options is None else options)
    validate(options)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in saved['model'].values()):
        raise ValueError('Gradient comparison requires FP32 checkpoint weights')
    model_name = saved['options']['model']
    current_config = settings[model_name + '_model']
    model_config = {key: value for key, value in saved['model_config'].items()
                    if key in current_config}
    dev = torch.device(device)
    model = build_model(model_name, model_config).to(
        device=dev, dtype=torch.float32, memory_format=torch.channels_last)
    load_core_weights(model, saved['model'])
    core_count = len(model.state_dict())
    ignored_count = len(saved['model']) - core_count
    cycle = saved['summary']['cycle']
    del saved
    sq = torch.arange(80, device=dev)
    indices = sq // 8 * 16 + 2 * (sq % 8) + (sq // 8 % 2)
    seeds = [((options['seed'] + cycle + offset) % 2**64) for offset in (1009, 65537)]
    samples = []
    positions = []
    cpu_backend = torch.backends.mkldnn.flags(enabled=False) if dev.type == 'cpu' else nullcontext()
    with cpu_backend:
        for index, seed in enumerate(seeds, 1):
            gradients, count = _gradient_samples(library, model, options, dev, seed, chunks, indices)
            samples.append(gradients)
            positions.append(count)
            print(f'Arena {index}: {count:,} positions, {chunks} gradient samples', flush=True)
    first, second = samples
    results = []
    group = 1
    while group <= chunks:
        a = first.reshape(chunks // group, group, -1).mean(1)
        b = second.reshape(chunks // group, group, -1).mean(1)
        dots = a @ b.T
        norms = a.norm(dim=1)[:, None] * b.norm(dim=1)[None, :]
        if (norms <= 0).any():
            raise RuntimeError('Zero gradient norm')
        cosine = dots / norms.clamp_min(1e-30)
        if not torch.isfinite(cosine).all():
            raise RuntimeError('Nonfinite gradient comparison')
        results.append(dict(microbatches=group,
                            perspectives=group * options['train_minibatch'],
                            cross_pairs=cosine.numel(),
                            mean_dot=dots.mean().item(),
                            mean_cosine=cosine.mean().item(),
                            median_cosine=cosine.median().item(),
                            positive_fraction=(dots > 0).float().mean().item()))
        group *= 2
    return dict(checkpoint=str(Path(checkpoint)), cycle=cycle, model=model_name,
                auxiliary_tensors_ignored=ignored_count,
                seeds=seeds, positions=positions, results=results)
