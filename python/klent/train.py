"""Sequential KLENT self-play / fitting / sampled old-versus-new evaluation."""
import argparse
import copy
import math
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
from config import CONFIG_PATH, settings
from klent.native import Arena, Batch
from klent import output
from models.factory import MODELS, build_model
MAX_PLY = 80


def validate(options):
    if type(options['cycles']) is not int or options['cycles'] < 0:
        raise ValueError('cycles must be a nonnegative integer')
    for key in ('n','m','train_minibatch','test_games','anchor_interval','test_interval','opponent_interval'):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f'klent.{key} must be a positive integer')
    if options['m'] < options['n']*MAX_PLY:
        raise ValueError('klent.m must be at least n*80')
    if options['test_games'] % 2:
        raise ValueError('test_games must be even for balanced sides')
    if options['train_minibatch'] % 2:
        raise ValueError('train_minibatch counts perspectives and must be even')
    if type(options.get('both_sides', False)) is not bool:
        raise ValueError('klent.both_sides must be a boolean')
    if type(options.get('fixed_opponent', False)) is not bool:
        raise ValueError('klent.fixed_opponent must be a boolean')
    if options.get('fixed_opponent', False) and options['n'] % 2:
        raise ValueError('fixed-opponent training requires an even klent.n')
    if type(options.get('compile_model', True)) is not bool:
        raise ValueError('klent.compile_model must be a boolean')
    if type(options.get('compile_max_autotune', False)) is not bool:
        raise ValueError('klent.compile_max_autotune must be a boolean')
    for key in ('alpha','beta','exploration_fraction','lambda','lr','weight_decay'):
        if not math.isfinite(options[key]) or options[key] < 0:
            raise ValueError(f'klent.{key} must be nonnegative and finite')
    if options['alpha']+options['beta'] <= 0 or options['exploration_fraction'] > 1 or options['lambda'] > 1 or options['lr'] <= 0:
        raise ValueError('require alpha+beta > 0, exploration_fraction, lambda values <= 1, and lr > 0')
    if type(options['seed']) is not int or not 0 <= options['seed'] < 2**64:
        raise ValueError('seed must fit u64')
    if options['model'] not in MODELS or options['optimizer'].lower() != 'adamw':
        raise ValueError('supported models: katago, katago_tf, resnet, maia; optimizer: adamw')
    references = options['reference_checkpoints']
    if (not isinstance(references, list) or
        any(not isinstance(path, str) or not path for path in references)):
        raise ValueError('klent.reference_checkpoints must be a list of paths')


def load_reference_model(path, device):
    """Build the fixed opponent with its own saved architecture."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    name = checkpoint.get('options', {}).get('model')
    if checkpoint.get('format_version') not in (1, 2) or name not in MODELS:
        raise ValueError(f'Unsupported reference checkpoint: {path}')
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in checkpoint['model'].values()):
        raise ValueError('Reference checkpoint requires FP32 weights')
    reference = MODELS[name](checkpoint['model_config']).to(
        device=device, dtype=torch.float32, memory_format=torch.channels_last)
    load_core_weights(reference, checkpoint['model'])
    return reference.eval().requires_grad_(False), checkpoint['summary']['cycle']


def losses(logits, q, target, actions, returns, valid):
    """Full 160-way CE, played-action MSE; padded rows have zero weight."""
    policy = -(target.float()*logits.flatten(1).float().log_softmax(1)).sum(1)
    value = (q.flatten(1).float().gather(1,actions[:,None]).squeeze(1)-returns).square()
    denominator = valid.sum().clamp_min(1)
    return (policy*valid).sum()/denominator, (value*valid).sum()/denominator


def load_core_weights(model, weights):
    """Load policy, value and tower weights from old seven-head checkpoints."""
    core = model.state_dict()
    source = {name: weights[name] for name in core}
    model.load_state_dict(source)


def restore_optimizer(optimizer, saved, saved_model, model):
    """Restore AdamW moments for retained parameters by name."""
    current = optimizer.state_dict()
    if len(saved['param_groups']) != len(current['param_groups']):
        raise ValueError('Checkpoint optimizer groups do not match the model')
    old_ids = [key for group in saved['param_groups'] for key in group['params']]
    old_names = list(saved_model)
    new_ids = [key for group in current['param_groups'] for key in group['params']]
    new_names = [name for name, _ in model.named_parameters()]
    if len(old_ids) != len(old_names) or len(new_ids) != len(new_names):
        raise ValueError('Checkpoint optimizer parameters cannot be matched to model weights')
    old_by_name = dict(zip(old_names, old_ids))
    for name, new_id in zip(new_names, new_ids):
        old_id = old_by_name.get(name)
        if old_id in saved['state']:
            current['state'][new_id] = saved['state'][old_id]
    for old_group, new_group in zip(saved['param_groups'], current['param_groups']):
        new_group.update({k: v for k, v in old_group.items() if k != 'params'})
    optimizer.load_state_dict(current)


def evaluation_rows(games):
    """CPU row indices matching native new-model result accounting."""
    if games < 2 or games % 2:
        raise ValueError('balanced evaluation requires a positive even game count')
    slot = torch.arange(games)
    new = 2*slot + (slot < games//2).long()
    return 2*slot + 1 - (new % 2), new


class ModelHistory:
    """Keep the previous model and every cycle-spaced anchor."""
    def __init__(self, interval, anchors=()):
        self.previous = None
        self.interval = interval
        self.anchors = list(anchors)

    @staticmethod
    def snapshot(model):
        return {k: v.detach().clone() for k,v in model.state_dict().items()}

    def remember_previous(self, model, cycle):
        self.previous = (cycle, self.snapshot(model))

    def initialize_anchors(self, model, cycle):
        if not self.anchors:
            self.anchors = [(cycle, self.snapshot(model))]

    def maybe_add_anchor(self, model, cycle):
        if cycle % self.interval == 0 and cycle > self.anchors[-1][0]:
            self.anchors.append((cycle, self.snapshot(model)))
            return cycle
        return None

    def opponents(self):
        if self.previous is not None:
            yield ('previous', *self.previous)
        for cycle, weights in self.anchors:
            yield ('anchor', cycle, weights)

    def checkpoint_anchors(self):
        return list(self.anchors)


class Device:
    def __init__(self, name):
        self.name = name
        self.cuda = torch.device(name).type == 'cuda'

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def transfer(self, tensors):
        return tuple(t.to(self.name, non_blocking=self.cuda) for t in tensors)

    def stamp(self):
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return perf_counter()

    def elapsed(self, first, second):
        return first.elapsed_time(second) if self.cuda else (second-first)*1000


def run(library, options=None, *, cycles=None, device='cuda', compile_model=None, checkpoint=None, resume=None, log_dir=None, fixed_opponent=None, references=None, opponent=None, max_seconds=None):
    """cycles=0 runs until interrupted or the Modal function timeout expires."""
    run_start = perf_counter()
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError('max_seconds must be positive')
    options = dict(settings['klent'] if options is None else options)
    if cycles is not None:
        options['cycles'] = cycles
    if fixed_opponent is not None:
        options['fixed_opponent'] = fixed_opponent
    if references is not None:
        options['reference_checkpoints'] = list(references)
    if opponent is not None:
        options['fixed_opponent'] = True
        options['opponent_checkpoint'] = str(opponent)
    restored = None
    if resume is not None:
        restored = torch.load(resume,map_location='cpu',weights_only=True)
        if restored['format_version'] not in (1, 2):
            raise ValueError('Unsupported checkpoint format')
        if any(t.is_floating_point() and t.dtype != torch.float32
               for t in restored['model'].values()):
            raise ValueError('Resume requires an FP32 checkpoint')
        # Architecture must match the weights; runtime training settings are current.
        if restored['options']['model'] not in MODELS:
            raise ValueError('Unsupported model in resume checkpoint')
        options['model'] = restored['options']['model']
        model_key = options['model']+'_model'
        model_config = {key: value for key, value in restored['model_config'].items()
                        if key in settings[model_key]}
        settings[model_key] = model_config
    validate(options)
    if compile_model is None:
        compile_model = options.get('compile_model', True)
    fixed_opponent = options['fixed_opponent']
    external_opponent = opponent is not None
    reference_paths = [Path(path) for path in options['reference_checkpoints']]
    reference_paths = [path if path.is_absolute() else CONFIG_PATH.parent / path
                       for path in reference_paths]
    cycles = options['cycles']
    dev = Device(device)
    torch.manual_seed(options['seed'])
    model = build_model(options['model']).to(device=device,dtype=torch.float32,
                                            memory_format=torch.channels_last)
    old = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=options['lr'],weight_decay=options['weight_decay'])
    if restored is not None:
        load_core_weights(model, restored['model'])
        restore_optimizer(optimizer, restored['optimizer'], restored['model'], model)
        # Loading AdamW also restores param-group settings: apply current config.
        for group in optimizer.param_groups:
            group['lr'] = options['lr']
            group['weight_decay'] = options['weight_decay']
        if options['seed'] == restored['options']['seed']:
            torch.set_rng_state(restored['torch_rng'])
            if dev.cuda and restored['cuda_rng']:
                torch.cuda.set_rng_state(restored['cuda_rng'][0],device=device)
        else:
            torch.manual_seed(options['seed'])
    with torch.random.fork_rng(devices=[]):
        reference_models = [(str(path), *load_reference_model(path, device)) for path in reference_paths]
        if opponent is not None:
            opponent_path = Path(opponent)
            if not opponent_path.is_absolute():
                opponent_path = CONFIG_PATH.parent / opponent_path
            opponent_checkpoint = torch.load(opponent_path, map_location='cpu', weights_only=True)
            if (opponent_checkpoint.get('format_version') not in (1, 2) or
                opponent_checkpoint.get('options', {}).get('model') not in MODELS):
                raise ValueError(f'Unsupported opponent checkpoint: {opponent_path}')
            if any(t.is_floating_point() and t.dtype != torch.float32
                   for t in opponent_checkpoint['model'].values()):
                raise ValueError('Opponent checkpoint requires FP32 weights')
            opponent_name = opponent_checkpoint['options']['model']
            opponent_config = opponent_checkpoint['model_config']
            opponent_model = MODELS[opponent_name](opponent_config).to(
                device=device, dtype=torch.float32, memory_format=torch.channels_last)
            load_core_weights(opponent_model, opponent_checkpoint['model'])
            opponent_model.eval().requires_grad_(False)
            training_opponent = (opponent_checkpoint['summary']['cycle'],
                                 ModelHistory.snapshot(opponent_model),
                                 opponent_name, opponent_config)
    compile_options = {'dynamic': False}
    if options.get('compile_max_autotune', False):
        compile_options['mode'] = 'max-autotune'
    network = torch.compile(model,**compile_options) if compile_model else model
    previous = torch.compile(old,**compile_options) if compile_model else old
    reference_networks = [(path, torch.compile(net,**compile_options) if compile_model else net, cycle)
                          for path, net, cycle in reference_models]
    opponent_network = (torch.compile(opponent_model, **compile_options) if compile_model else opponent_model
                        ) if external_opponent else None
    sq = torch.arange(80,device=device)
    indices = sq//8*16+2*(sq%8)+(sq//8%2)
    one_side = fixed_opponent and not options.get('both_sides', False)
    batch = Batch(options['train_minibatch']//2 if one_side else options['train_minibatch'],
                  pinned=dev.cuda, perspectives=1 if one_side else 2)
    rows = torch.arange(batch.rows,device=device)
    if fixed_opponent:
        game_rows = torch.arange(options['n'])
        current_rows = 2*game_rows + (game_rows >= options['n']//2).long()
        fixed_rows = 2*game_rows + 1 - (current_rows % 2)
    completed = 0 if restored is None else restored['summary']['cycle']
    stop_cycle = completed + cycles
    summaries = []
    history = ModelHistory(options['anchor_interval'])
    history.initialize_anchors(model, completed)
    if not external_opponent:
        training_opponent = (completed, history.snapshot(model)) if fixed_opponent else None

    def infer(net, boards):
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            b, = dev.transfer((boards,))
            pi,q = net(b.contiguous(memory_format=torch.channels_last))
            # Blocking CPU copies also prevent reuse of compiled output storage
            # before the native consumer finishes with the result.
            return (pi.flatten(1)[:,indices].float().cpu().contiguous(),
                    q.flatten(1)[:,indices].float().cpu().contiguous())

    def match(opponent):
        with Arena(library,options,evaluation=True,seed=(options['seed']+completed+1)%(2**64),pinned=dev.cuda) as test:
            # First half: new P1. Second half: new P0. Each network sees
            # test_games rows, including zero rows for finished slots.
            old_rows, new_rows = evaluation_rows(options['test_games'])
            while True:
                boards = test.inputs()
                torch.compiler.cudagraph_mark_step_begin()
                p0,q0 = infer(opponent,boards[old_rows])
                p1,q1 = infer(network,boards[new_rows])
                pi = torch.empty((2*test.n,80))
                q = torch.empty_like(pi)
                pi[old_rows], pi[new_rows] = p0, p1
                q[old_rows], q[new_rows] = q0, q1
                if test.step(pi,q):
                    break
            stats = test.stats()
            return stats.wins,stats.draws,stats.losses

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        metadata = dict(started_at=datetime.now(timezone.utc).isoformat(),
                        resume=str(resume) if resume else None, initial_cycle=completed,
                        options=options, model_config=settings[options['model']+'_model'],
                        precision='fp32-weights-bf16-autocast', device=str(device),
                        gpu=torch.cuda.get_device_name(device) if dev.cuda else None,
                        training_mode='fixed_opponent' if fixed_opponent else 'self_play',
                        references=[str(path) for path in reference_paths],
                        torch_version=str(torch.__version__), compiled=compile_model,
                        compile_mode=('max-autotune' if options.get('compile_max_autotune', False)
                                      else 'default') if compile_model else None)
        (log_dir/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    output.setup(options, device, compile_model)
    if restored is None and checkpoint is not None:
        initial = dict(cycle=0, evaluations=[], anchor_cycles=[0],
                       training_mode='fixed_opponent' if fixed_opponent else 'self_play',
                       training_opponent_cycle=training_opponent[0] if fixed_opponent else None,
                       next_training_opponent_cycle=training_opponent[0] if fixed_opponent else None,
                       timestamp=datetime.now(timezone.utc).isoformat())
        checkpoint(model, optimizer, options, initial, history.checkpoint_anchors(),
                   training_opponent)
    with Arena(library, options, pinned=dev.cuda, fixed_opponent=fixed_opponent,
               seed=(options['seed']+completed)%(2**64)) as arena:
        output.emit(f'  Setup      {perf_counter()-run_start:.2f} s')
        model_time = cpu_time = 0.0
        arena_step = 0
        while cycles == 0 or completed < stop_cycle:
            if arena_step == 0:
                cycle_start = perf_counter()
                if fixed_opponent:
                    played_opponent_cycle = training_opponent[0]
                    if not external_opponent:
                        load_core_weights(old, training_opponent[1])
            arena_step += 1
            t = perf_counter()
            boards = arena.inputs()
            encoding_time = perf_counter()-t
            model.eval()
            torch.compiler.cudagraph_mark_step_begin()
            dev.sync(); t = perf_counter()
            if fixed_opponent:
                if options.get('both_sides', False):
                    target_pi,target_q = infer(network,boards)
                    pi,q = target_pi.clone(),target_q.clone()
                else:
                    current_pi,current_q = infer(network,boards[current_rows])
                    pi = torch.empty((2*arena.n,80))
                    q = torch.empty_like(pi)
                    pi[current_rows],q[current_rows] = current_pi,current_q
                fixed_pi,fixed_q = infer(opponent_network if external_opponent else previous,
                                         boards[fixed_rows])
                pi[fixed_rows],q[fixed_rows] = fixed_pi,fixed_q
            else:
                pi,q = infer(network,boards)
            dev.sync(); model_ms = (perf_counter()-t)*1000
            model_time += model_ms/1000
            t = perf_counter()
            full = (arena.step(pi,q,target_logits=target_pi,target_q=target_q)
                    if fixed_opponent and options.get('both_sides', False) else arena.step(pi,q))
            if not full:
                cpu_ms = (perf_counter()-t+encoding_time)*1000
                cpu_time += cpu_ms/1000
                continue
            count = arena.stats().positions
            dropped = arena.reset()
            cpu_time += perf_counter()-t+encoding_time
            processing_start = perf_counter()
            arena.shuffle()
            shuffle_time = perf_counter()-processing_start
            dev.sync(); training_start = perf_counter()
            history.remember_previous(model, completed)
            model.train()
            loss_sum = policy_sum = value_sum = 0.0
            timing_sum = [0.0, 0.0, 0.0]
            for start in range(0,count,batch.positions):
                valid_positions = arena.batch(start,batch)
                b,stored_target,actions,returns = dev.transfer(batch.tensors)
                b = b.contiguous(memory_format=torch.channels_last)
                valid = (rows < batch.perspectives*valid_positions).float()
                optimizer.zero_grad(set_to_none=True)
                torch.compiler.cudagraph_mark_step_begin()
                forward_start = dev.stamp()
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                    logits,values = network(b)
                    policy_loss,value_loss = losses(logits,values,stored_target,actions,returns,valid)
                    loss = policy_loss + value_loss
                forward_end = dev.stamp()
                loss.backward()
                backward_end = dev.stamp()
                optimizer.step()
                optimizer_end = dev.stamp()
                dev.sync()
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    raise RuntimeError('Nonfinite training loss')
                loss_sum += loss_value*valid_positions
                batch_number = start//batch.positions+1
                policy_sum += policy_loss.item()*valid_positions
                value_sum += value_loss.item()*valid_positions
                for j, (first, second) in enumerate(((forward_start,forward_end),
                        (forward_end,backward_end),(backward_end,optimizer_end))):
                    timing_sum[j] += dev.elapsed(first,second)
            optimizer_steps = batch_number
            arena.clear()
            dev.sync(); training_time = perf_counter()-training_start
            training_exclusive_time = training_time
            cpu_total = cpu_time + shuffle_time
            model.eval()
            evaluations = []
            test_time = 0.0
            next_cycle = completed + 1
            if next_cycle % options['test_interval'] == 0:
                test_start = perf_counter()
                matched_cycles = {}
                for kind, opponent_cycle, weights in history.opponents():
                    opponent_start = perf_counter()
                    shared_match = opponent_cycle in matched_cycles
                    if shared_match:
                        wins,draws,losses_count = matched_cycles[opponent_cycle]
                    else:
                        load_core_weights(old, weights)
                        wins,draws,losses_count = match(previous)
                        matched_cycles[opponent_cycle] = wins,draws,losses_count
                    dev.sync()
                    evaluations.append(dict(kind=kind, is_anchor=kind == 'anchor',
                        age=next_cycle-opponent_cycle, opponent_cycle=opponent_cycle,
                        wins=wins,draws=draws,losses=losses_count,
                        win_rate=wins/options['test_games'],
                        score_rate=(wins+0.5*draws)/options['test_games'],
                        seconds=perf_counter()-opponent_start,
                        shared_match=shared_match))
                for path, opponent, opponent_cycle in reference_networks:
                    opponent_start = perf_counter()
                    wins,draws,losses_count = match(opponent)
                    dev.sync()
                    evaluations.append(dict(kind='reference', is_anchor=False,
                        reference_path=path, age=None, opponent_cycle=opponent_cycle,
                        wins=wins,draws=draws,losses=losses_count,
                        win_rate=wins/options['test_games'],
                        score_rate=(wins+0.5*draws)/options['test_games'],
                        seconds=perf_counter()-opponent_start, shared_match=False))
                dev.sync(); test_time = perf_counter()-test_start
            latest = evaluations[0] if evaluations else None
            wins,draws,losses_count = ((latest[k] for k in ('wins','draws','losses'))
                                        if latest else (None,None,None))
            completed += 1
            anchor_cycle = history.maybe_add_anchor(model, completed)
            anchor_updates = [dict(new_cycle=anchor_cycle)] if anchor_cycle is not None else []
            if fixed_opponent and not external_opponent and completed % options['opponent_interval'] == 0:
                training_opponent = (completed, history.snapshot(model))
            summary = dict(cycle=completed,evaluations=evaluations,states=count,dropped_states=dropped,loss=loss_sum/count,
                           training_mode='fixed_opponent' if fixed_opponent else 'self_play',
                           training_opponent_cycle=played_opponent_cycle if fixed_opponent else None,
                           next_training_opponent_cycle=training_opponent[0] if fixed_opponent else None,
                           anchor_cycles=[cycle for cycle,_ in history.anchors],
                           anchor_updates=anchor_updates,
                           model_seconds=model_time,cpu_seconds=cpu_total,
                           shuffle_seconds=shuffle_time,
                           training_seconds=training_exclusive_time,
                           strength_test_seconds=test_time,wins=wins,draws=draws,losses=losses_count,
                           selfplay_steps=arena_step,batches=batch_number,
                           optimizer_steps=optimizer_steps,
                           mean_cpu_ms=cpu_total*1000/arena_step,
                           mean_model_ms=model_time*1000/arena_step,
                           mean_forward_ms=timing_sum[0]/batch_number,
                           mean_backward_ms=timing_sum[1]/batch_number,
                           mean_optimizer_ms=timing_sum[2]/optimizer_steps,
                           policy_loss=policy_sum/count,q_loss=value_sum/count)
            summary.update(timestamp=datetime.now(timezone.utc).isoformat(),
                           elapsed_seconds=perf_counter()-run_start,
                           win_rate=latest['win_rate'] if latest else None,
                           score_rate=latest['score_rate'] if latest else None)
            if log_dir is not None:
                with (log_dir/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(summary, allow_nan=False)+'\n')
            if checkpoint is not None:
                checkpoint(model, optimizer, options, summary, history.checkpoint_anchors(),
                           training_opponent)
            if cycles:
                summaries.append(summary)
            # A single report cannot include the duration of its own print call.
            cycle_time = perf_counter()-cycle_start
            timing = dict(total=cycle_time, cpu=cpu_total, selfplay_model=model_time,
                          training_model=training_exclusive_time, strength_test=test_time,
                          other=cycle_time-cpu_total-model_time-training_exclusive_time-test_time)
            output.summary(summary, timing)
            arena_step = 0
            model_time = cpu_time = 0.0
            if max_seconds is not None and perf_counter()-run_start >= max_seconds:
                break
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,default=Path(__file__).resolve().parents[2]/'rust/target/release/libalpha_lines_game.so')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--no-compile',action='store_true',help='CPU smoke testing only')
    parser.add_argument('--resume',type=Path,help='Restore weights, optimizer state, architecture and cycle number')
    parser.add_argument('--reference',action='append',default=None,
                        help='Strength-test checkpoint path; repeat for multiple references')
    parser.add_argument('--oponent','--opponent',dest='opponent',type=Path,
                        help='Train against this exact checkpoint for every cycle')
    parser.add_argument('--name', help='Run folder name under checkpoints/klent and logs/klent')
    parser.add_argument('--checkpoint-dir',type=Path,help='Output directory; defaults to a new local run')
    parser.add_argument('--log-dir',type=Path,help='Metrics directory; defaults to logs/klent/<run-id>')
    args = parser.parse_args()
    from uuid import uuid4
    import re
    from klent.checkpoint import save
    if args.name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}', args.name):
        parser.error('--name must be 1–100 characters: letters, digits, dots, _ or -')
    run_id = args.name or uuid4().hex
    directory = args.checkpoint_dir or Path('checkpoints/klent')/run_id
    log_dir = args.log_dir or Path('logs/klent')/run_id
    if args.name and ((args.checkpoint_dir is None and directory.exists()) or
                      (args.log_dir is None and log_dir.exists())):
        parser.error(f'Run folder already exists: {run_id}')
    def checkpoint(model,optimizer,options,summary,anchors,training_opponent):
        path = save(directory,model,optimizer,options,summary,anchors,training_opponent)
        output.emit(f'Checkpoint saved: {path}')
    run(args.library,device=args.device,compile_model=False if args.no_compile else None,
        checkpoint=checkpoint,resume=args.resume,references=args.reference,
        opponent=args.opponent,log_dir=log_dir)


if __name__ == '__main__':
    main()
