"""Sequential KLENT self-play / fitting / sampled old-versus-new evaluation."""
import argparse
import copy
import math
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from config import CONFIG_PATH, settings
from klent.native import Arena, Batch
from klent import output
from models.factory import MODELS, build_model
from models.katago import KataGoNet
MAX_PLY = 80


def validate(options):
    if type(options['cycles']) is not int or options['cycles'] < 0:
        raise ValueError('cycles must be a nonnegative integer')
    for key in ('n','m','train_minibatch','test_games'):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f'klent.{key} must be a positive integer')
    if options['m'] < options['n']*MAX_PLY:
        raise ValueError('klent.m must be at least n*80')
    if options['n'] % 8:
        raise ValueError('klent.n must be divisible by eight for balanced opponent partitions')
    if options['test_games'] % 2:
        raise ValueError('test_games must be even for balanced sides')
    if options['train_minibatch'] % 2:
        raise ValueError('train_minibatch counts perspectives and must be even')
    if type(options['policy_recalculation']) is not bool:
        raise ValueError('klent.policy_recalculation must be a boolean')
    for key in ('alpha','beta','exploration_fraction','lambda','lr','weight_decay',
                'policy_loss_weight','q_loss_weight'):
        if not math.isfinite(options[key]) or options[key] < 0:
            raise ValueError(f'klent.{key} must be nonnegative and finite')
    if options['alpha']+options['beta'] <= 0 or options['exploration_fraction'] > 1 or options['lambda'] > 1 or options['lr'] <= 0:
        raise ValueError('require alpha+beta > 0, exploration_fraction, lambda values <= 1, and lr > 0')
    if type(options['seed']) is not int or not 0 <= options['seed'] < 2**64:
        raise ValueError('seed must fit u64')
    if options['model'] not in MODELS or options['optimizer'].lower() != 'adamw':
        raise ValueError('supported models: katago, katago_tf, resnet, maia; optimizer: adamw')
    if not isinstance(options['reference_checkpoint'], str) or not options['reference_checkpoint']:
        raise ValueError('klent.reference_checkpoint must be a nonempty path')
    thresholds = options['anchor_thresholds']
    if (not isinstance(thresholds, list) or len(thresholds) != 3 or
        any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1
            for value in thresholds) or
        any(a >= b for a,b in zip(thresholds,thresholds[1:]))):
        raise ValueError('klent.anchor_thresholds must be three increasing score rates in (0, 1]')


def load_reference_model(path, device):
    """Build the fixed opponent with its own saved architecture."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint.get('options', {}).get('model') != 'katago':
        raise ValueError(f'Unsupported reference checkpoint: {path}')
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in checkpoint['model'].values()):
        raise ValueError('Reference checkpoint requires FP32 weights')
    reference = KataGoNet(checkpoint['model_config']).to(
        device=device, dtype=torch.float32, memory_format=torch.channels_last)
    load_core_weights(reference, checkpoint['model'])
    return reference.eval().requires_grad_(False), checkpoint['summary']['cycle']


def losses(logits, q, target, actions, returns, valid):
    """Full 160-way CE, played-action MSE; padded rows have zero weight."""
    policy = -(target.float()*logits.flatten(1).float().log_softmax(1)).sum(1)
    value = (q.flatten(1).float().gather(1,actions[:,None]).squeeze(1)-returns).square()
    denominator = valid.sum().clamp_min(1)
    return (policy*valid).sum()/denominator, (value*valid).sum()/denominator


def recalculated_policy(logits, q, boards, valid, alpha, beta):
    """Recreate KLENT's legal softmax from the current minibatch forward pass."""
    empty = boards[:,1] > 0.5
    opening = empty.flatten(1).sum(1) == 80
    player = torch.arange(boards.shape[0],device=boards.device) % 2
    columns = torch.arange(16,device=boards.device).view(1,1,16)
    opening_half = torch.where(player[:,None,None] == 0,columns < 8,columns >= 8)
    legal = (empty & (~opening[:,None,None] | opening_half)).flatten(1)
    # Padded rows have no legal squares; give softmax one temporary finite logit.
    fallback = F.one_hot(torch.zeros(boards.shape[0],device=boards.device,dtype=torch.long),160).bool()
    safe_legal = torch.where(valid[:,None] > 0,legal,fallback)
    scores = (q.flatten(1).float() + beta*logits.flatten(1).float()) / (alpha+beta)
    policy = scores.masked_fill(~safe_legal,float('-inf')).softmax(1)
    return torch.where(valid[:,None] > 0,policy,torch.zeros_like(policy)).detach()


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


def historical_cycles(cycle):
    """Geometric opponents at distances d and 2d, up to 64 cycles old."""
    if cycle < 1:
        raise ValueError('cycle must be positive')
    distance = min(32, 1 << (max(1, cycle//2).bit_length()-1))
    return max(0, cycle-distance), max(0, cycle-2*distance)


def historical_rows(games, quarter):
    """Side-2 rows for one 256-game quarter of a 1024-game arena."""
    if games % 8 or quarter not in (2, 3):
        raise ValueError('historical partitions require a multiple of eight games')
    slots = torch.arange(quarter*games//4, (quarter+1)*games//4)
    return 2*slots + (slots >= (2*quarter+1)*games//8).long()


def historical_file(directory, cycle):
    name = 'cycle-000000-weights.pt' if cycle == 0 else f'cycle-{cycle:06d}.pt'
    return Path(directory)/name


def load_historical_window(resume, completed, model):
    """Restore the CPU model snapshots needed for the next 64 cycles."""
    recent = {completed: ModelHistory.snapshot(model)}
    for cycle in range(max(0, completed-64), completed):
        path = historical_file(Path(resume).parent, cycle)
        if not path.is_file():
            raise FileNotFoundError(f'Missing historical opponent for resume: {path}')
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if cycle and saved.get('summary', {}).get('cycle') != cycle:
            raise ValueError(f'Historical checkpoint has the wrong cycle: {path}')
        weights = saved if cycle == 0 else saved['model']
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in weights.values()):
            raise ValueError(f'Historical opponent requires FP32 weights: {path}')
        recent[cycle] = {name: weights[name] for name in model.state_dict()}
    return recent


class ModelHistory:
    """Keep the previous model and three score-gated moving anchors."""
    def __init__(self, thresholds=(0.70, 0.80, 0.90), anchors=()):
        self.previous = None
        self.thresholds = tuple(thresholds)
        self.anchors = list(anchors)
        if self.anchors and len(self.anchors) != 3:
            raise ValueError('Expected three saved KLENT anchors')

    @staticmethod
    def snapshot(model):
        return {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}

    def remember_previous(self, model, cycle):
        self.previous = (cycle, self.snapshot(model))

    def initialize_anchors(self, model, cycle):
        if not self.anchors:
            weights = self.snapshot(model)
            self.anchors = [(cycle, weights)] * 3

    def update_anchors(self, model, cycle, evaluations):
        updates = []
        weights = None
        for result in evaluations:
            index = result.get('anchor_index')
            if index is None or result['score_rate'] <= self.thresholds[index-1]:
                continue
            if weights is None:
                weights = self.snapshot(model)
            old_cycle = self.anchors[index-1][0]
            self.anchors[index-1] = (cycle, weights)
            updates.append(dict(anchor=index, old_cycle=old_cycle, new_cycle=cycle,
                                score_rate=result['score_rate']))
        return updates

    def opponents(self):
        if self.previous is not None:
            yield ('previous', *self.previous, None)
        for index, (cycle, weights) in enumerate(self.anchors, 1):
            yield ('anchor', cycle, weights, index)

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


def run(library, options=None, *, cycles=None, device='cuda', compile_model=True, checkpoint=None, resume=None, log_dir=None, history_dir=None):
    """cycles=0 runs until interrupted or the Modal function timeout expires."""
    run_start = perf_counter()
    options = dict(settings['klent'] if options is None else options)
    if cycles is not None:
        options['cycles'] = cycles
    restored = None
    saved_anchors = None
    if resume is not None:
        restored = torch.load(resume,map_location='cpu',weights_only=True)
        if restored['format_version'] != 1:
            raise ValueError('Unsupported checkpoint format')
        # Legacy fixed-duration anchors have different semantics and cannot be
        # promoted by score. New checkpoints carry the moving anchors in full.
        if restored.get('anchor_system') == 'score_gated_v1':
            saved_anchors = restored['anchors']
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
    reference_path = Path(options['reference_checkpoint'])
    if not reference_path.is_absolute():
        reference_path = CONFIG_PATH.parent / reference_path
    cycles = options['cycles']
    dev = Device(device)
    torch.manual_seed(options['seed'])
    model = build_model(options['model']).to(device=device,dtype=torch.float32,
                                            memory_format=torch.channels_last)
    old = copy.deepcopy(model).eval().requires_grad_(False)
    second_old = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=options['lr'],weight_decay=options['weight_decay'])
    if restored is not None:
        load_core_weights(model, restored['model'])
        if saved_anchors is not None:
            core_names = model.state_dict().keys()
            saved_anchors = [(cycle, {name: weights[name] for name in core_names})
                             for cycle, weights in saved_anchors]
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
    reference_model, reference_cycle = load_reference_model(reference_path, device)
    network = torch.compile(model,dynamic=False) if compile_model else model
    previous = torch.compile(old,dynamic=False) if compile_model else old
    second_previous = torch.compile(second_old,dynamic=False) if compile_model else second_old
    reference_network = torch.compile(reference_model,dynamic=False) if compile_model else reference_model
    sq = torch.arange(80,device=device)
    indices = sq//8*16+2*(sq%8)+(sq//8%2)
    batch = Batch(options['train_minibatch'],pinned=dev.cuda)
    rows = torch.arange(batch.rows,device=device)
    completed = 0 if restored is None else restored['summary']['cycle']
    stop_cycle = completed + cycles
    summaries = []
    history = ModelHistory(options['anchor_thresholds'], saved_anchors or ())
    if restored is not None and saved_anchors is None:
        history.initialize_anchors(model, completed)

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
                        torch_version=str(torch.__version__), compiled=compile_model)
        (log_dir/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    output.setup(options, device, compile_model)
    recent = (load_historical_window(resume, completed, model) if resume is not None
              else {0: ModelHistory.snapshot(model)})
    if history_dir is not None:
        history_dir = Path(history_dir)
        history_dir.mkdir(parents=True, exist_ok=True)
        if resume is None:
            path = historical_file(history_dir,0)
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite initial historical model: {path}')
            temporary = path.with_suffix('.pt.tmp')
            torch.save(recent[0], temporary)
            temporary.replace(path)
        elif history_dir.resolve() != Path(resume).parent.resolve():
            for cycle in range(max(0, completed-64), completed+1):
                source = Path(resume) if cycle == completed else historical_file(Path(resume).parent, cycle)
                target = historical_file(history_dir, cycle)
                if target.exists():
                    raise FileExistsError(f'Refusing to overwrite historical model: {target}')
                try:
                    os.link(source,target)
                except OSError:
                    shutil.copy2(source,target)
    with Arena(library, options, pinned=dev.cuda, seed=(options['seed']+completed)%(2**64)) as arena:
        output.emit(f'  Setup      {perf_counter()-run_start:.2f} s')
        historical_indices = (historical_rows(arena.n,2),historical_rows(arena.n,3))
        model_time = cpu_time = 0.0
        arena_step = 0
        while cycles == 0 or completed < stop_cycle:
            if arena_step == 0:
                cycle_start = perf_counter()
                opponent_cycles = historical_cycles(completed+1)
                for opponent_cycle, opponent_model in zip(opponent_cycles,(old,second_old)):
                    if opponent_cycle != completed:
                        load_core_weights(opponent_model,recent[opponent_cycle])
            arena_step += 1
            t = perf_counter()
            boards = arena.inputs()
            encoding_time = perf_counter()-t
            model.eval()
            torch.compiler.cudagraph_mark_step_begin()
            dev.sync(); t = perf_counter()
            pi,q = infer(network,boards)
            opponent_pi,opponent_q = pi.clone(),q.clone()
            for selected, opponent_cycle, opponent_net in zip(
                    historical_indices,opponent_cycles,(previous,second_previous)):
                if opponent_cycle == completed:
                    continue
                opponent_pi[selected],opponent_q[selected] = infer(opponent_net,boards[selected])
            dev.sync(); model_ms = (perf_counter()-t)*1000
            model_time += model_ms/1000
            t = perf_counter()
            full = arena.step(pi,q,opponent_pi,opponent_q)
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
            for start in range(0,count,batch.rows//2):
                valid_positions = arena.batch(start,batch)
                b,stored_target,actions,returns = dev.transfer(batch.tensors)
                b = b.contiguous(memory_format=torch.channels_last)
                valid = (rows < 2*valid_positions).float()
                optimizer.zero_grad(set_to_none=True)
                torch.compiler.cudagraph_mark_step_begin()
                forward_start = dev.stamp()
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                    logits,values = network(b)
                    target = (recalculated_policy(logits,values,b,valid,options['alpha'],options['beta'])
                              if options['policy_recalculation'] else stored_target)
                    policy_loss,value_loss = losses(logits,values,target,actions,returns,valid)
                    loss = (options['policy_loss_weight'] * policy_loss
                            + options['q_loss_weight'] * value_loss)
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
                batch_number = start//(batch.rows//2)+1
                policy_sum += policy_loss.item()*valid_positions
                value_sum += value_loss.item()*valid_positions
                for j, (first, second) in enumerate(((forward_start,forward_end),
                        (forward_end,backward_end),(backward_end,optimizer_end))):
                    timing_sum[j] += dev.elapsed(first,second)
            arena.clear()
            dev.sync(); training_time = perf_counter()-training_start
            training_exclusive_time = training_time
            cpu_total = cpu_time + shuffle_time
            test_start = perf_counter()
            model.eval()
            evaluations = []
            # The first three anchors initially share cycle 1. Reuse that
            # deterministic match while keeping one report row per threshold.
            matched_cycles = {}
            opponents = list(history.opponents())
            opponents.append(('reference', reference_cycle, None, None))
            for kind, opponent_cycle, weights, anchor_index in opponents:
                opponent_start = perf_counter()
                shared_match = kind != 'reference' and opponent_cycle in matched_cycles
                if shared_match:
                    wins,draws,losses_count = matched_cycles[opponent_cycle]
                else:
                    if kind == 'reference':
                        opponent = reference_network
                    else:
                        # Reuse one compiled opponent for previous and anchors.
                        load_core_weights(old, weights)
                        opponent = previous
                    wins,draws,losses_count = match(opponent)
                    if kind != 'reference':
                        matched_cycles[opponent_cycle] = wins,draws,losses_count
                dev.sync()
                evaluations.append(dict(kind=kind, anchor_index=anchor_index,
                    is_anchor=anchor_index is not None,
                    age=None if kind == 'reference' else completed+1-opponent_cycle,
                    opponent_cycle=opponent_cycle,
                    wins=wins,draws=draws,losses=losses_count,
                    win_rate=wins/options['test_games'],
                    score_rate=(wins+0.5*draws)/options['test_games'],
                    seconds=perf_counter()-opponent_start,
                    shared_match=shared_match))
            dev.sync(); test_time = perf_counter()-test_start
            latest = evaluations[0]
            wins,draws,losses_count = (latest[k] for k in ('wins','draws','losses'))
            completed += 1
            recent[completed] = ModelHistory.snapshot(model)
            for old_cycle in list(recent):
                if old_cycle < completed-64:
                    del recent[old_cycle]
            if not history.anchors:
                history.initialize_anchors(model, completed)
                anchor_updates = []
            else:
                anchor_updates = history.update_anchors(model, completed, evaluations)
            summary = dict(cycle=completed,evaluations=evaluations,states=count,dropped_states=dropped,loss=loss_sum/count,
                           training_opponent_cycles=list(opponent_cycles),
                           anchor_cycles=[cycle for cycle,_ in history.anchors],
                           anchor_updates=anchor_updates,
                           model_seconds=model_time,cpu_seconds=cpu_total,
                           shuffle_seconds=shuffle_time,
                           training_seconds=training_exclusive_time,
                           strength_test_seconds=test_time,wins=wins,draws=draws,losses=losses_count,
                           selfplay_steps=arena_step,batches=batch_number,
                           mean_cpu_ms=cpu_total*1000/arena_step,
                           mean_model_ms=model_time*1000/arena_step,
                           mean_forward_ms=timing_sum[0]/batch_number,
                           mean_backward_ms=timing_sum[1]/batch_number,
                           mean_optimizer_ms=timing_sum[2]/batch_number,
                           policy_loss=policy_sum/count,q_loss=value_sum/count,
                           policy_loss_weight=options['policy_loss_weight'],
                           q_loss_weight=options['q_loss_weight'])
            summary.update(timestamp=datetime.now(timezone.utc).isoformat(),
                           elapsed_seconds=perf_counter()-run_start,
                           win_rate=latest['win_rate'], score_rate=latest['score_rate'])
            if checkpoint is not None:
                checkpoint(model, optimizer, options, summary, history.checkpoint_anchors())
            if log_dir is not None:
                with (log_dir/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(summary, allow_nan=False)+'\n')
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
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,default=Path(__file__).resolve().parents[2]/'rust/target/release/libalpha_lines_game.so')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--no-compile',action='store_true',help='CPU smoke testing only')
    parser.add_argument('--resume',type=Path,help='Restore weights, optimizer state, architecture and cycle number')
    parser.add_argument('--checkpoint-dir',type=Path,help='Output directory; defaults to a new local run')
    parser.add_argument('--log-dir',type=Path,help='Metrics directory; defaults to logs/klent/<run-id>')
    args = parser.parse_args()
    from uuid import uuid4
    from klent.checkpoint import save
    run_id = uuid4().hex
    directory = args.checkpoint_dir or Path('checkpoints/klent')/run_id
    log_dir = args.log_dir or Path('logs/klent')/run_id
    def checkpoint(model,optimizer,options,summary,anchors):
        path = save(directory,model,optimizer,options,summary,anchors)
        output.emit(f'Checkpoint saved: {path}')
    run(args.library,device=args.device,compile_model=not args.no_compile,
        checkpoint=checkpoint,resume=args.resume,log_dir=log_dir,history_dir=directory)


if __name__ == '__main__':
    main()
