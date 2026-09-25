"""PUCT self-search continuation of a KLENT KataGo checkpoint."""
import argparse
import copy
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch

from config import settings
from klent.native import Arena as EvaluationArena
from klent.train import Device, ModelHistory, evaluation_rows
from mcts.native import Arena, Batch, default_node_capacity
from models.katago import KataGoNet


def validate(o):
    for key in ('g', 'b', 't', 's', 'steps', 'node_capacity', 'train_minibatch', 'test_games'):
        if type(o[key]) is not int or o[key] < 1:
            raise ValueError(f'mcts.{key} must be a positive integer')
    if type(o['cycles']) is not int or o['cycles'] < 0:
        raise ValueError('mcts.cycles must be a nonnegative integer')
    if o['g'] > 65535 or o['b'] > o['g'] or o['t'] > o['g']:
        raise ValueError('require g <= 65535 and b,t <= g')
    if o['s'] > 2**32 - 1:
        raise ValueError('mcts.s must fit u32')
    if type(o['node_capacity_factor']) is not int or o['node_capacity_factor'] < 1:
        raise ValueError('mcts.node_capacity_factor must be a positive integer')
    if o['train_minibatch'] % 2 or o['test_games'] % 2:
        raise ValueError('train_minibatch and test_games must be even')
    for key in ('c_puct', 'dirichlet_alpha', 'dirichlet_epsilon', 'lr',
                'weight_decay', 'policy_loss_weight', 'q_loss_weight'):
        if not math.isfinite(o[key]) or o[key] < 0:
            raise ValueError(f'mcts.{key} must be finite and nonnegative')
    if min(o['c_puct'], o['dirichlet_alpha'], o['lr']) <= 0 or o['dirichlet_epsilon'] > 1:
        raise ValueError('c_puct, dirichlet_alpha, lr must be positive; epsilon <= 1')
    for key in ('double_flip_augment', 'all_flip_augment'):
        if type(o[key]) is not bool:
            raise ValueError(f'mcts.{key} must be a boolean')
    if o['q_visit_weighting'] not in ('equal', 'visits'):
        raise ValueError('mcts.q_visit_weighting must be equal or visits')
    if (not isinstance(o['anchor_after'], list) or len(o['anchor_after']) != 2 or
        any(type(x) is not int or x < 1 for x in o['anchor_after']) or
        o['anchor_after'][0] >= o['anchor_after'][1]):
        raise ValueError('mcts.anchor_after must be two increasing positive cycle numbers')
    if type(o['seed']) is not int or not 0 <= o['seed'] < 2**64:
        raise ValueError('mcts.seed must fit u64')
    if o['model'] != 'katago' or o['optimizer'].lower() != 'adamw':
        raise ValueError('supported model: katago; optimizer: adamw')


def losses(logits, q, policy_target, q_target, visits, weighting='equal'):
    """CE per perspective and masked MSE over visited action values."""
    policy = -(policy_target.float() * logits.flatten(1).float().log_softmax(1)).sum(1).mean()
    mask = (visits > 0).float() if weighting == 'equal' else visits.float()
    value = ((q.flatten(1).float() - q_target.float()).square() * mask).sum() / mask.sum().clamp_min(1)
    return policy, value


def augmentation_mode(options):
    if options['all_flip_augment']:
        return 'all'
    if options['double_flip_augment']:
        return 'double'
    return 'off'


def transforms_for(positions, double_flip_augment=False, all_flip_augment=False):
    variants = (0, 1, 2, 3) if all_flip_augment else (0, 3) if double_flip_augment else (0,)
    ids = torch.arange(positions, dtype=torch.int64).repeat_interleave(len(variants)).to(torch.uint32)
    transforms = torch.tensor(variants, dtype=torch.uint8).repeat(positions)
    order = torch.randperm(ids.numel())
    return ids[order].contiguous(), transforms[order].contiguous()


class History(ModelHistory):
    def __init__(self, anchor_after, start_cycle):
        super().__init__(start_cycle=start_cycle)
        self.anchor_after = tuple(anchor_after)

    def remember_anchor(self, model, cycle):
        if cycle - self.start_cycle in self.anchor_after:
            self.anchors.append((cycle, self.snapshot(model)))


def save(directory, model, optimizer, options, summary, source_cycle, baseline_model, anchors):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"cycle-{summary['cycle']:06d}.pt"
    if path.exists():
        raise FileExistsError(path)
    for cycle, weights in anchors:
        anchor = directory / 'anchors' / f'anchor-{cycle:06d}.pt'
        if not anchor.exists():
            anchor.parent.mkdir(parents=True, exist_ok=True)
            temporary = anchor.with_suffix('.pt.tmp')
            torch.save({k: v.detach().cpu() for k, v in weights.items()}, temporary)
            temporary.replace(anchor)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(dict(format_version=2, trainer='mcts', model=model.state_dict(),
        optimizer=optimizer.state_dict(), options=dict(options),
        model_config=dict(settings['katago_model']), summary=dict(summary),
        source_klent_cycle=source_cycle,
        baseline_model={k: v.detach().cpu() for k, v in baseline_model.items()},
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        exact_resume=False), temporary)
    temporary.replace(path)
    return path


def report(row, timing):
    loss_label_width = 14
    lines = ['', f"MCTS cycle {row['cycle']} complete", '-' * 54,
        f"  Source KLENT cycle          {row['source_klent_cycle']}",
        f"  Positions                   {row['source_positions']:,}",
        f"  Augmented positions         {row['training_positions']:,}",
        f"  Perspective rows            {row['training_rows']:,}",
        f"  Search steps                {row['steps']:,}",
        f"  Mean MCTS step              {row['mean_mcts_step_seconds']:.2f} s",
        f"  Training minibatches        {row['minibatches']:,}",
        f"  Loss                        {row['loss']:.5f}",
        f"    {'Policy':<{loss_label_width}}{row['policy_loss']:>9.5f} × {row['policy_loss_weight']:g}",
        f"    {'Action value':<{loss_label_width}}{row['q_loss']:>9.5f} × {row['q_loss_weight']:g}",
        f"    {'Q weighting':<{loss_label_width}}{row['q_visit_weighting']:>9}",
        f"  Search calls                {row['model_calls']:,}",
        f"  Evaluated nodes             {row['evaluated_nodes']:,}",
        f"  Model batch fill            {row['buffer_fill']:.1%}",
        f"  Completed games             {row['finished_games']:,}",
        f"  Live nodes at cycle end     {row['live_nodes']:,} / {row['node_capacity']:,}"
        f" ({row['live_nodes']/row['node_capacity']:.1%})",
        f"  Node allocation failures    {row['node_exhaustion']:,}", '']
    for label, key in [('Cycle total', 'total'), ('Search CPU', 'search_cpu'),
        ('Search model', 'search_model'), ('Training preparation', 'train_prep'),
        ('Training model', 'train_model'), ('Strength test', 'strength_test'),
        ('Checkpoint', 'checkpoint_log'), ('Other', 'other')]:
        lines.append(f'  {label:<26}{timing[key]:>10.2f} s')
    if row['initial_setup_seconds']:
        lines.append(f"  {'Initial setup (outside cycle)':<26}{row['initial_setup_seconds']:>10.2f} s")
        lines.append(f"  {'  of which compile/warmup':<26}{row['compile_warmup_seconds']:>10.2f} s")
    lines.extend(('', '  Opponent                  W     D     L    Score'))
    for result in row['evaluations']:
        w, d, l = (result[k] for k in ('wins', 'draws', 'losses'))
        label = result['label']
        lines.append(f'  {label:<24}{w:5d} {d:5d} {l:5d}  {(w+0.5*d)/(w+d+l):6.1%}')
    print('\n'.join(lines), flush=True)


def run(library, options=None, *, cycles=None, device='cuda', compile_model=True,
        resume='checkpoints/resume.pt', checkpoint_dir=None, log_dir=None):
    started = perf_counter()
    options = dict(settings['mcts'] if options is None else options)
    if cycles is not None:
        options['cycles'] = cycles
    capacity_derived = options.get('node_capacity') is None
    if capacity_derived:
        options['node_capacity'] = default_node_capacity(options['g'], options['s'],
                                                         options['node_capacity_factor'])
    validate(options)
    restored = torch.load(resume, map_location='cpu', weights_only=True)
    if restored['format_version'] not in (1, 2):
        raise ValueError('Unsupported checkpoint format')
    if restored['options']['model'] != 'katago':
        raise ValueError('Only KataGo checkpoints are supported')
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in restored['model'].values()):
        raise ValueError('Source model must have FP32 weights')
    mcts_resume = restored['format_version'] == 2
    if mcts_resume and restored.get('trainer') != 'mcts':
        raise ValueError('Unsupported format-2 checkpoint')
    source_cycle = restored['source_klent_cycle'] if mcts_resume else restored['summary']['cycle']
    completed = restored['summary']['cycle'] if mcts_resume else 0
    baseline_model = restored['baseline_model'] if mcts_resume else restored['model']
    model_config = dict(restored['model_config'])
    for key, value in settings['katago_model'].items():
        model_config.setdefault(key, value)
    settings['katago_model'] = model_config
    torch.manual_seed(options['seed'])
    dev = Device(device)
    model = KataGoNet().to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
    missing, unexpected = model.load_state_dict(restored['model'], strict=False)
    allowed_missing = ('mark_class_head.', 'immediate_score_head.', 'discounted_score_head.',
                       'discounted_mark_head.', 'opponent_policy_head.')
    if any(not key.startswith(allowed_missing) for key in missing) or any(not key.startswith(('mark_class_head.', 'score_embed.')) for key in unexpected):
        raise ValueError(f'Checkpoint model mismatch: missing={missing}, unexpected={unexpected}')
    old = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'], weight_decay=options['weight_decay'])
    if mcts_resume:
        optimizer.load_state_dict(restored['optimizer'])
        for group in optimizer.param_groups:
            group['lr'], group['weight_decay'] = options['lr'], options['weight_decay']
        if options['seed'] == restored['options']['seed']:
            torch.set_rng_state(restored['torch_rng'])
            if dev.cuda and restored['cuda_rng']:
                torch.cuda.set_rng_state(restored['cuda_rng'][0], device=device)
    network = torch.compile(model, dynamic=False) if compile_model else model
    previous = torch.compile(old, dynamic=False) if compile_model else old
    sq = torch.arange(80, device=device)
    indices = sq // 8 * 16 + 2 * (sq % 8) + (sq // 8 % 2)
    batch = Batch(options['train_minibatch'] // 2, pinned=dev.cuda)
    history = History(options['anchor_after'], completed)
    history.anchors.append((0, {k: v.detach().clone() for k, v in baseline_model.items()}))
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir/'run.json').write_text(json.dumps(dict(started_at=datetime.now(timezone.utc).isoformat(),
            resume=str(resume), source_klent_cycle=source_cycle, initial_mcts_cycle=completed,
            options=options, model_config=model_config, precision='fp32-weights-bf16-autocast',
            device=str(device), compiled=compile_model), indent=2)+'\n')

    def infer(net, boards):
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=dev.cuda):
            b, = dev.transfer((boards,))
            pi, q, *_ = net(b.contiguous(memory_format=torch.channels_last))
            return (pi.flatten(1)[:,indices].float().cpu().contiguous(),
                    q.flatten(1)[:,indices].float().cpu().contiguous())

    # torch.compile is lazy. Exercise each inference shape and the training
    # backward graph before the first cycle so compilation does not inflate its
    # search, training, or strength-test timings. No optimizer step is taken.
    warmup_seconds = 0.0
    if compile_model:
        print('  Compiling search, training, and evaluation model graphs...', flush=True)
        warmup_start = perf_counter()
        search_boards = torch.zeros((2*options['b'], 5, 10, 16), dtype=torch.float32)
        test_boards = torch.zeros((options['test_games'], 5, 10, 16), dtype=torch.float32)
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            infer(network, search_boards)
            torch.compiler.cudagraph_mark_step_begin()
            infer(network, test_boards)
            torch.compiler.cudagraph_mark_step_begin()
            infer(previous, test_boards)
        del search_boards, test_boards
        model.train()
        train_rows = options['train_minibatch']
        train_boards = torch.zeros((train_rows, 5, 10, 16), device=device,
                                   dtype=torch.float32).contiguous(memory_format=torch.channels_last)
        target = torch.full((train_rows, 160), 1/160, device=device)
        zeros = torch.zeros_like(target)
        visited = torch.ones_like(target)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=dev.cuda):
                logits, q_pred, *_ = network(train_boards)
                policy_loss, q_loss = losses(logits, q_pred, target, zeros, visited,
                                             options['q_visit_weighting'])
                loss = options['policy_loss_weight']*policy_loss + options['q_loss_weight']*q_loss
            loss.backward()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        dev.sync()
        warmup_seconds = perf_counter() - warmup_start
        print(f'  Model compile/warmup {warmup_seconds:.2f} s', flush=True)

    def evaluate(weights, cycle):
        old.load_state_dict(weights, strict=False)
        with EvaluationArena(library, settings['klent'] | {'test_games': options['test_games']},
                             evaluation=True, seed=(options['seed']+cycle) % 2**64,
                             pinned=dev.cuda) as test:
            old_rows, new_rows = evaluation_rows(options['test_games'])
            while True:
                boards = test.inputs()
                torch.compiler.cudagraph_mark_step_begin()
                p0, q0 = infer(previous, boards[old_rows])
                p1, q1 = infer(network, boards[new_rows])
                pi = torch.empty((2*test.n,80), dtype=torch.float32)
                q = torch.empty_like(pi)
                pi[old_rows], pi[new_rows] = p0, p1
                q[old_rows], q[new_rows] = q0, q1
                if test.step(pi, q):
                    break
            stats = test.stats()
            return stats.wins, stats.draws, stats.losses

    capacity_label = f"g*s*{options['node_capacity_factor']}" if capacity_derived else 'override'
    print(f'MCTS | KataGo | source KLENT cycle {source_cycle} | {device}\n'
          f'  g={options["g"]} b={options["b"]} t={options["t"]} s={options["s"]} steps={options["steps"]}\n'
          f'  node capacity={options["node_capacity"]:,}'
          f' ({capacity_label})\n'
          f'  augment={augmentation_mode(options)} minibatch={options["train_minibatch"]} '
          f'lr={options["lr"]:g} decay={options["weight_decay"]:g}', flush=True)
    summaries = []
    initial_completed = completed
    stop_cycle = completed + options['cycles']
    with Arena(library, options, seed=(options['seed']+completed) % 2**64, pinned=dev.cuda) as arena:
        setup_seconds = perf_counter()-started
        print(f'  Setup {setup_seconds:.2f} s (includes model compile/warmup)', flush=True)
        while options['cycles'] == 0 or completed < stop_cycle:
            cycle_start = perf_counter()
            search_cpu = search_model = 0.0
            step_seconds = []
            model.eval()
            for step_index in range(options['steps']):
                step_start = perf_counter()
                model_before_step = search_model
                before = arena.stats()
                while True:
                    boards, _ = arena.inputs()
                    torch.compiler.cudagraph_mark_step_begin()
                    dev.sync(); t = perf_counter()
                    pi, q = infer(network, boards)
                    dev.sync(); search_model += perf_counter() - t
                    stepped = arena.advance(pi, q)
                    if stepped:
                        break
                step_elapsed = perf_counter() - step_start
                search_cpu += step_elapsed - (search_model - model_before_step)
                step_seconds.append(step_elapsed)
                after = arena.stats()
                projected_search = sum(step_seconds) / len(step_seconds) * options['steps']
                print(f"  MCTS step {step_index+1}/{options['steps']}  {step_elapsed:.2f} s"
                      f"  |  calls {after.model_calls-before.model_calls:,}"
                      f"  |  evaluated {after.evaluated-before.evaluated:,}"
                      f"  |  collection estimate {projected_search:.1f} s", flush=True)
            stats = arena.stats()
            if stats.records != options['steps'] * options['t']:
                raise RuntimeError('MCTS produced the wrong number of training positions')
            history.remember_previous(model, completed)
            t = perf_counter()
            ids, transforms = transforms_for(stats.records, options['double_flip_augment'],
                                             options['all_flip_augment'])
            prep_time = perf_counter() - t
            model.train()
            policy_sum = value_sum = total_sum = 0.0
            train_model = 0.0
            minibatch_positions = batch.positions
            for start in range(0, ids.numel(), minibatch_positions):
                end = min(start + minibatch_positions, ids.numel())
                t = perf_counter()
                count = arena.batch(ids[start:end], transforms[start:end], batch)
                b, policy_target, q_target, visited = dev.transfer(tensor[:2*count] for tensor in batch.tensors)
                b = b.contiguous(memory_format=torch.channels_last)
                dev.sync(); prep_time += perf_counter() - t
                optimizer.zero_grad(set_to_none=True)
                torch.compiler.cudagraph_mark_step_begin()
                dev.sync(); t = perf_counter()
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=dev.cuda):
                    logits, q_pred, *_ = network(b)
                    policy_loss, q_loss = losses(logits, q_pred, policy_target, q_target, visited,
                                                 options['q_visit_weighting'])
                    loss = options['policy_loss_weight']*policy_loss + options['q_loss_weight']*q_loss
                loss.backward()
                optimizer.step()
                dev.sync(); train_model += perf_counter() - t
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    raise RuntimeError('Nonfinite MCTS training loss')
                total_sum += loss_value * count
                policy_sum += policy_loss.item() * count
                value_sum += q_loss.item() * count
            training_positions = ids.numel()
            model.eval()
            test_start = perf_counter()
            evaluations = []
            opponents = [('previous', completed, history.previous[1])]
            opponents.extend((f'anchor {cycle}', cycle, weights) for cycle, weights in history.anchors)
            tested = {}
            for label, opponent_cycle, weights in opponents:
                if opponent_cycle in tested:
                    evaluations[tested[opponent_cycle]]['label'] += ' / ' + label
                    continue
                wins, draws, losses_count = evaluate(weights, completed+1)
                evaluations.append(dict(label=label, opponent_cycle=opponent_cycle,
                    wins=wins, draws=draws, losses=losses_count,
                    score_rate=(wins+0.5*draws)/options['test_games']))
                tested[opponent_cycle] = len(evaluations) - 1
            dev.sync(); test_time = perf_counter() - test_start
            completed += 1
            history.remember_anchor(model, completed)
            row = dict(cycle=completed, source_klent_cycle=source_cycle,
                initial_setup_seconds=setup_seconds if completed == initial_completed + 1 else 0.0,
                compile_warmup_seconds=warmup_seconds if completed == initial_completed + 1 else 0.0,
                source_positions=stats.records, training_positions=training_positions,
                training_rows=2*training_positions,
                steps=options['steps'], augmentation=augmentation_mode(options),
                model_calls=stats.model_calls,
                evaluated_nodes=stats.evaluated, buffer_fill=stats.evaluated/max(1,stats.model_calls*options['b']),
                finished_games=stats.finished_games, live_nodes=stats.live_nodes,
                node_capacity=options['node_capacity'],
                node_exhaustion=stats.exhausted, minibatches=math.ceil(training_positions/minibatch_positions),
                mcts_step_seconds=step_seconds,
                mean_mcts_step_seconds=sum(step_seconds)/len(step_seconds),
                loss=total_sum/training_positions, policy_loss=policy_sum/training_positions,
                q_loss=value_sum/training_positions,
                q_visit_weighting=options['q_visit_weighting'],
                policy_loss_weight=options['policy_loss_weight'], q_loss_weight=options['q_loss_weight'],
                search_seconds=search_cpu+search_model,
                search_cpu_seconds=search_cpu, search_model_seconds=search_model,
                train_prep_seconds=prep_time, train_model_seconds=train_model,
                strength_test_seconds=test_time, evaluations=evaluations,
                timestamp=datetime.now(timezone.utc).isoformat())
            checkpoint_start = perf_counter()
            if checkpoint_dir is not None:
                path = save(checkpoint_dir, model, optimizer, options, row, source_cycle,
                            baseline_model,
                            history.checkpoint_anchors())
                print(f'Checkpoint saved: {path}', flush=True)
            checkpoint_log = perf_counter() - checkpoint_start
            total = perf_counter() - cycle_start
            timing = dict(total=total, search_cpu=search_cpu, search_model=search_model,
                train_prep=prep_time, train_model=train_model, strength_test=test_time,
                checkpoint_log=checkpoint_log,
                other=total-search_cpu-search_model-prep_time-train_model-test_time-checkpoint_log)
            row.update(total_seconds=total, checkpoint_seconds=checkpoint_log,
                       other_seconds=timing['other'])
            if log_dir is not None:
                with (log_dir/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
            report(row, timing)
            arena.network_update()
            if options['cycles']:
                summaries.append(row)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=Path(__file__).resolve().parents[2]/'rust/target/release/libalpha_lines_game.so')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--resume', type=Path, default=Path('checkpoints/resume.pt'))
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--log-dir', type=Path)
    args = parser.parse_args()
    run_id = uuid4().hex
    run(args.library, device=args.device, compile_model=not args.no_compile,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir or Path('checkpoints/mcts')/run_id,
        log_dir=args.log_dir or Path('logs/mcts')/run_id)


if __name__ == '__main__':
    main()
