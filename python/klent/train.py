"""Sequential KLENT self-play / fitting / sampled old-versus-new evaluation."""
import argparse
import copy
import math
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from config import settings
from klent.native import Arena, Batch
from klent import output, checkpoint as checkpoint_io
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
    if options['test_games'] % 2:
        raise ValueError('test_games must be even for balanced sides')
    if options['train_minibatch'] % 2:
        raise ValueError('train_minibatch counts perspectives and must be even')
    for key in ('alpha','beta','exploration_fraction','lambda','lr','weight_decay',
                'policy_loss_weight','opponent_policy_weight','q_loss_weight','mark_class_loss_weight',
                'immediate_score_weight','discounted_score_weight','discounted_score_lambda',
                'discounted_mark_weight','discounted_mark_lambda'):
        if not math.isfinite(options[key]) or options[key] < 0:
            raise ValueError(f'klent.{key} must be nonnegative and finite')
    if options['alpha']+options['beta'] <= 0 or options['exploration_fraction'] > 1 or options['lambda'] > 1 or options['discounted_score_lambda'] > 1 or options['discounted_mark_lambda'] > 1 or options['lr'] <= 0:
        raise ValueError('require alpha+beta > 0, exploration_fraction, lambda values <= 1, and lr > 0')
    if type(options['seed']) is not int or not 0 <= options['seed'] < 2**64:
        raise ValueError('seed must fit u64')
    if options['model'] != 'katago' or options['optimizer'].lower() != 'adamw':
        raise ValueError('supported model: katago; optimizer: adamw')


def losses(logits, q, target, actions, returns, valid):
    """Full 160-way CE, played-action MSE; padded rows have zero weight."""
    policy = -(target.float()*logits.flatten(1).float().log_softmax(1)).sum(1)
    value = (q.flatten(1).float().gather(1,actions[:,None]).squeeze(1)-returns).square()
    denominator = valid.sum().clamp_min(1)
    return (policy*valid).sum()/denominator, (value*valid).sum()/denominator


def opponent_policy_loss(logits, target, valid):
    """Predict the other player's improved policy from each paired board row."""
    other = target.reshape(-1, 2, 160).flip(1).reshape(-1, 160).float()
    per_row = -(other * logits.flatten(1).float().log_softmax(1)).sum(1)
    return (per_row * valid).sum() / valid.sum().clamp_min(1)


def mark_class_targets(classes, indices):
    """Expand perspective-relative classes into six spatial one-hot planes."""
    present = classes >= 0
    target = torch.zeros((classes.shape[0], 6, 160), device=classes.device, dtype=torch.float32)
    target[:, :, indices] = F.one_hot(classes.long().clamp_min(0), 6).permute(0, 2, 1).float() * present[:, None, :]
    return target.reshape(-1, 6, 10, 16)


def mark_class_loss(logits, target, valid):
    per_cell = -(target * logits.float().log_softmax(1)).sum(1)
    occupied = target.sum(1) * valid[:, None, None]
    return (per_cell * occupied).sum() / occupied.sum().clamp_min(1)


def immediate_score_loss(logits, scores, valid):
    """Cross-entropy for both perspective-relative scores; ignore padded rows."""
    targets = scores.long()
    per_player = F.cross_entropy(logits.float().flatten(0, 1),
                                 targets.clamp(0, 80).flatten(), reduction='none')
    per_row = per_player.reshape(-1, 2).mean(1)
    return (per_row * valid).sum() / valid.sum().clamp_min(1)


def discounted_score_loss(logits, target, valid):
    """Soft-target cross-entropy for both players' future-score distributions."""
    target = target.float()
    target = target / target.sum(-1, keepdim=True).clamp_min(1)
    per_row = -(target * logits.float().log_softmax(-1)).sum(-1).mean(-1)
    return (per_row * valid).sum() / valid.sum().clamp_min(1)


def discounted_mark_targets(distributions, indices):
    """Expand eight-class per-square future targets into spatial planes."""
    target = torch.zeros((distributions.shape[0], 8, 160), device=distributions.device, dtype=torch.float32)
    target[:, :, indices] = distributions.float().permute(0, 2, 1)
    return target.reshape(-1, 8, 10, 16)


def discounted_mark_loss(logits, target, valid):
    """Soft-target CE over all 80 playable squares, ignoring padded rows."""
    target = target / target.sum(1, keepdim=True).clamp_min(1)
    per_cell = -(target * logits.float().log_softmax(1)).sum(1)
    mask = target.sum(1) * valid[:, None, None]
    return (per_cell * mask).sum() / mask.sum().clamp_min(1)


def restore_optimizer(optimizer, saved, saved_model, model):
    """Restore moments by parameter name when heads are added or score input removed."""
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
    """Keep the previous model and up to three fixed eight-cycle anchors."""
    def __init__(self, anchors=()):
        self.previous = None
        self.anchors = [(cycle, {k: v.clone() for k,v in weights.items()})
                        for cycle, weights in anchors]

    @staticmethod
    def snapshot(model):
        return {k: v.detach().clone() for k,v in model.state_dict().items()}

    def remember_previous(self, model, cycle):
        self.previous = (cycle, self.snapshot(model))

    def remember_initial_anchor(self, model, cycle):
        if not self.anchors:
            self.anchors.append((cycle, self.snapshot(model)))

    def remember_anchor(self, model, cycle):
        first_cycle = self.anchors[0][0] if self.anchors else 0
        if cycle > first_cycle and (cycle-first_cycle) % 8 == 0 and len(self.anchors) < 3:
            self.anchors.append((cycle, self.snapshot(model)))

    def opponents(self):
        if self.previous is not None:
            yield ('previous', *self.previous)
        for cycle, weights in self.anchors:
            if self.previous is None or cycle != self.previous[0]:
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


def run(library, options=None, *, cycles=None, device='cuda', compile_model=True, checkpoint=None, resume=None, anchor_dir=None, log_dir=None):
    """cycles=0 runs until interrupted or the Modal function timeout expires."""
    run_start = perf_counter()
    options = dict(settings['klent'] if options is None else options)
    if cycles is not None:
        options['cycles'] = cycles
    restored = None
    if resume is not None:
        restored = torch.load(resume,map_location='cpu',weights_only=True)
        if restored['format_version'] != 1:
            raise ValueError('Unsupported checkpoint format')
        if any(t.is_floating_point() and t.dtype != torch.float32
               for t in restored['model'].values()):
            raise ValueError('Resume requires an FP32 checkpoint')
        # Architecture must match the weights; runtime training settings are current.
        if restored['options']['model'] != 'katago':
            raise ValueError('Only KataGo checkpoints are supported')
        options['model'] = 'katago'
        model_key = options['model']+'_model'
        model_config = dict(restored['model_config'])
        model_config.setdefault('immediate_score_filters', settings[model_key]['immediate_score_filters'])
        model_config.setdefault('discounted_score_filters', settings[model_key]['discounted_score_filters'])
        model_config.setdefault('discounted_mark_filters', settings[model_key]['discounted_mark_filters'])
        model_config.setdefault('opponent_policy_filters', settings[model_key]['opponent_policy_filters'])
        settings[model_key] = model_config
    validate(options)
    cycles = options['cycles']
    dev = Device(device)
    torch.manual_seed(options['seed'])
    model = KataGoNet().to(device=device,dtype=torch.float32,memory_format=torch.channels_last)
    old = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=options['lr'],weight_decay=options['weight_decay'])
    if restored is not None:
        missing, unexpected = model.load_state_dict(restored['model'], strict=False)
        if any(not key.startswith(('mark_class_head.', 'score_embed.')) for key in unexpected) or any(not key.startswith(('mark_class_head.', 'immediate_score_head.', 'discounted_score_head.', 'discounted_mark_head.', 'opponent_policy_head.')) for key in missing):
            raise ValueError(f'Checkpoint model mismatch: missing={missing}, unexpected={unexpected}')
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
    network = torch.compile(model,dynamic=False) if compile_model else model
    previous = torch.compile(old,dynamic=False) if compile_model else old
    sq = torch.arange(80,device=device)
    indices = sq//8*16+2*(sq%8)+(sq//8%2)
    batch = Batch(options['train_minibatch'],pinned=dev.cuda)
    rows = torch.arange(batch.rows,device=device)
    completed = 0 if restored is None else restored['summary']['cycle']
    stop_cycle = completed + cycles
    summaries = []
    history = ModelHistory(checkpoint_io.load_anchors(resume, restored, anchor_dir)
                           if restored is not None else ())
    if restored is not None:
        history.remember_initial_anchor(model, completed)

    def infer(net, boards):
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            b, = dev.transfer((boards,))
            pi,q,_,_,_,_,_ = net(b.contiguous(memory_format=torch.channels_last))
            # Blocking CPU copies also prevent reuse of compiled output storage
            # before the native consumer finishes with the result.
            return (pi.flatten(1)[:,indices].float().cpu().contiguous(),
                    q.flatten(1)[:,indices].float().cpu().contiguous())

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
    with Arena(library, options, pinned=dev.cuda, seed=(options['seed']+completed)%(2**64)) as arena:
        output.emit(f'  Setup      {perf_counter()-run_start:.2f} s')
        model_time = cpu_time = 0.0
        arena_step = 0
        while cycles == 0 or completed < stop_cycle:
            if arena_step == 0:
                cycle_start = perf_counter()
            arena_step += 1
            t = perf_counter()
            boards = arena.inputs()
            encoding_time = perf_counter()-t
            model.eval()
            torch.compiler.cudagraph_mark_step_begin()
            dev.sync(); t = perf_counter()
            pi,q = infer(network,boards)
            dev.sync(); model_ms = (perf_counter()-t)*1000
            model_time += model_ms/1000
            t = perf_counter()
            full = arena.step(pi,q)
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
            loss_sum = policy_sum = opponent_policy_sum = value_sum = mark_sum = immediate_score_sum = discounted_score_sum = discounted_mark_sum = 0.0
            scoring_processing_time = 0.0
            timing_sum = [0.0, 0.0, 0.0]
            for start in range(0,count,batch.rows//2):
                valid_positions = arena.batch(start,batch)
                b,s,target,actions,returns,classes,discounted_scores,discounted_marks = dev.transfer(batch.tensors)
                dev.sync(); scoring_processing_start = perf_counter()
                class_target = mark_class_targets(classes, indices)
                future_mark_target = discounted_mark_targets(discounted_marks, indices)
                dev.sync()
                scoring_processing_time += perf_counter()-scoring_processing_start
                b = b.contiguous(memory_format=torch.channels_last)
                valid = (rows < 2*valid_positions).float()
                optimizer.zero_grad(set_to_none=True)
                torch.compiler.cudagraph_mark_step_begin()
                forward_start = dev.stamp()
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                    logits,values,mark_logits,score_logits,discounted_logits,discounted_mark_logits,opponent_logits = network(b)
                    policy_loss,value_loss = losses(logits,values,target,actions,returns,valid)
                    other_policy_loss = opponent_policy_loss(opponent_logits,target,valid)
                    mark_loss = mark_class_loss(mark_logits,class_target,valid)
                    score_loss = immediate_score_loss(score_logits,s,valid)
                    future_score_loss = discounted_score_loss(discounted_logits,discounted_scores,valid)
                    future_mark_loss = discounted_mark_loss(discounted_mark_logits,future_mark_target,valid)
                    loss = (options['policy_loss_weight'] * policy_loss
                            + options['opponent_policy_weight'] * other_policy_loss
                            + options['q_loss_weight'] * value_loss
                            + options['mark_class_loss_weight'] * mark_loss
                            + options['immediate_score_weight'] * score_loss
                            + options['discounted_score_weight'] * future_score_loss
                            + options['discounted_mark_weight'] * future_mark_loss)
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
                opponent_policy_sum += other_policy_loss.item()*valid_positions
                value_sum += value_loss.item()*valid_positions
                mark_sum += mark_loss.item()*valid_positions
                immediate_score_sum += score_loss.item()*valid_positions
                discounted_score_sum += future_score_loss.item()*valid_positions
                discounted_mark_sum += future_mark_loss.item()*valid_positions
                for j, (first, second) in enumerate(((forward_start,forward_end),
                        (forward_end,backward_end),(backward_end,optimizer_end))):
                    timing_sum[j] += dev.elapsed(first,second)
            arena.clear()
            dev.sync(); training_time = perf_counter()-training_start
            training_exclusive_time = training_time-scoring_processing_time
            cpu_total = cpu_time + shuffle_time + scoring_processing_time
            test_start = perf_counter()
            model.eval()
            evaluations = []
            for kind, opponent_cycle, weights in history.opponents():
                opponent_start = perf_counter()
                # Reuse one compiled opponent and copy into its existing tensors.
                old.load_state_dict(weights)
                with Arena(library,options,evaluation=True,seed=(options['seed']+completed+1)%(2**64),pinned=dev.cuda) as test:
                    # First half: new P1. Second half: new P0. Each network still
                    # sees test_games rows, including zero rows for finished slots.
                    old_rows, new_rows = evaluation_rows(options['test_games'])
                    while True:
                        b = test.inputs()
                        torch.compiler.cudagraph_mark_step_begin()
                        p0,q0 = infer(previous,b[old_rows])
                        p1,q1 = infer(network,b[new_rows])
                        pi = torch.empty((2*test.n,80))
                        q = torch.empty_like(pi)
                        pi[old_rows], pi[new_rows] = p0, p1
                        q[old_rows], q[new_rows] = q0, q1
                        if test.step(pi,q):
                            break
                    stats = test.stats()
                    wins,draws,losses_count = stats.wins,stats.draws,stats.losses
                dev.sync()
                evaluations.append(dict(kind=kind,
                    is_anchor=any(cycle == opponent_cycle for cycle,_ in history.anchors),
                    age=completed+1-opponent_cycle,
                    opponent_cycle=opponent_cycle,
                    wins=wins,draws=draws,losses=losses_count,
                    win_rate=wins/options['test_games'],
                    score_rate=(wins+0.5*draws)/options['test_games'],
                    seconds=perf_counter()-opponent_start))
            dev.sync(); test_time = perf_counter()-test_start
            latest = evaluations[0]
            wins,draws,losses_count = (latest[k] for k in ('wins','draws','losses'))
            completed += 1
            history.remember_anchor(model, completed)
            summary = dict(cycle=completed,evaluations=evaluations,states=count,dropped_states=dropped,loss=loss_sum/count,
                           model_seconds=model_time,cpu_seconds=cpu_total,
                           shuffle_seconds=shuffle_time,scoring_head_processing_seconds=scoring_processing_time,
                           training_seconds=training_exclusive_time,
                           strength_test_seconds=test_time,wins=wins,draws=draws,losses=losses_count,
                           selfplay_steps=arena_step,batches=batch_number,
                           mean_cpu_ms=cpu_total*1000/arena_step,
                           mean_model_ms=model_time*1000/arena_step,
                           mean_forward_ms=timing_sum[0]/batch_number,
                           mean_backward_ms=timing_sum[1]/batch_number,
                           mean_optimizer_ms=timing_sum[2]/batch_number,
                           policy_loss=policy_sum/count,opponent_policy_loss=opponent_policy_sum/count,
                           q_loss=value_sum/count,mark_class_loss=mark_sum/count,
                           immediate_score_loss=immediate_score_sum/count,
                           discounted_score_loss=discounted_score_sum/count,
                           discounted_mark_loss=discounted_mark_sum/count,
                           policy_loss_weight=options['policy_loss_weight'],
                           opponent_policy_weight=options['opponent_policy_weight'],
                           q_loss_weight=options['q_loss_weight'],
                           mark_class_loss_weight=options['mark_class_loss_weight'],
                           immediate_score_weight=options['immediate_score_weight'],
                           discounted_score_weight=options['discounted_score_weight'],
                           discounted_mark_weight=options['discounted_mark_weight'])
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
    parser.add_argument('--anchor-dir',type=Path,help='Directory containing anchors/ when resuming a copied checkpoint')
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
        checkpoint=checkpoint,resume=args.resume,anchor_dir=args.anchor_dir,log_dir=log_dir)


if __name__ == '__main__':
    main()
