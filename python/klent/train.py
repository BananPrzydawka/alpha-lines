"""Sequential KLENT self-play / fitting / sampled old-versus-new evaluation."""
import argparse
from collections import deque
import copy
import math
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
from config import settings
from klent.native import Arena, Batch
from klent import output
from models.katago import KataGoNet
from models.resnet import ResNet


def validate(options):
    for key in ('n','m','max_ply','train_minibatch','test_games'):
        if type(options[key]) is not int or options[key] < 1:
            raise ValueError(f'klent.{key} must be a positive integer')
    if options['max_ply'] != 80:
        raise ValueError('klent.max_ply must be the game storage bound, 80')
    if options['m'] < options['n']*options['max_ply']:
        raise ValueError('klent.m must be at least n*max_ply')
    if options['test_games'] % 2:
        raise ValueError('test_games must be even for balanced sides')
    if options['train_minibatch'] % 2:
        raise ValueError('train_minibatch counts perspectives and must be even')
    for key in ('alpha','beta','lambda','lr','weight_decay'):
        if not math.isfinite(options[key]) or options[key] < 0:
            raise ValueError(f'klent.{key} must be nonnegative and finite')
    if options['alpha']+options['beta'] <= 0 or options['lambda'] > 1 or options['lr'] <= 0:
        raise ValueError('require alpha+beta > 0, lambda <= 1, and lr > 0')
    if type(options['seed']) is not int or not 0 <= options['seed'] < 2**64:
        raise ValueError('seed must fit u64')
    if options['model'] not in ('resnet','katago') or options['optimizer'].lower() != 'adamw':
        raise ValueError('supported models: resnet, katago; optimizer: adamw')


def losses(logits, q, target, actions, returns, valid):
    """Full 160-way CE, played-action MSE; padded rows have zero weight."""
    policy = -(target.float()*logits.flatten(1).float().log_softmax(1)).sum(1)
    value = (q.flatten(1).float().gather(1,actions[:,None]).squeeze(1)-returns).square()
    denominator = valid.sum().clamp_min(1)
    return (policy*valid).sum()/denominator, (value*valid).sum()/denominator


def evaluation_rows(games):
    """CPU row indices matching native new-model result accounting."""
    if games < 2 or games % 2:
        raise ValueError('balanced evaluation requires a positive even game count')
    slot = torch.arange(games)
    new = 2*slot + (slot < games//2).long()
    return 2*slot + 1 - (new % 2), new


class ModelHistory:
    """Eight independent CPU weight snapshots, newest first when selecting ages."""
    def __init__(self):
        self.snapshots = deque(maxlen=8)

    def remember(self, model):
        self.snapshots.append({k: v.detach().to('cpu', copy=True)
                               for k,v in model.state_dict().items()})

    def opponents(self):
        return [(age,self.snapshots[-age]) for age in (1,2,4,8)
                if age <= len(self.snapshots)]


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


def run(library, options=None, *, cycles=None, device='cuda', compile_model=True, checkpoint=None, resume=None, log_dir=None):
    """cycles=0 runs until interrupted or the Modal function timeout expires."""
    options = dict(settings['klent'] if options is None else options)
    cycles = options['cycles'] if cycles is None else cycles
    restored = None
    if resume is not None:
        restored = torch.load(resume,map_location='cpu',weights_only=True)
        if restored['format_version'] != 1:
            raise ValueError('Unsupported checkpoint format')
        if any(t.is_floating_point() and t.dtype != torch.float32
               for t in restored['model'].values()):
            raise ValueError('Resume requires an FP32 checkpoint')
        # Architecture must match the weights; runtime training settings are current.
        options['model'] = restored['options']['model']
        settings[options['model']+'_model'] = dict(restored['model_config'])
    options['cycles'] = cycles
    validate(options)
    if type(cycles) is not int or cycles < 0:
        raise ValueError('cycles must be nonnegative')
    dev = Device(device)
    torch.set_num_threads(1)
    torch.manual_seed(options['seed'])
    factory = {'katago':KataGoNet,'resnet':ResNet}[options['model']]
    model = factory().to(device=device,dtype=torch.float32,memory_format=torch.channels_last)
    old = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=options['lr'],weight_decay=options['weight_decay'])
    if restored is not None:
        model.load_state_dict(restored['model'])
        optimizer.load_state_dict(restored['optimizer'])
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
    network = torch.compile(model,mode='max-autotune',dynamic=False) if compile_model else model
    previous = torch.compile(old,mode='max-autotune',dynamic=False) if compile_model else old
    sq = torch.arange(80,device=device)
    indices = sq//8*16+2*(sq%8)+(sq//8%2)
    batch = Batch(options['train_minibatch'],pinned=dev.cuda)
    rows = torch.arange(batch.rows,device=device)
    completed = 0 if restored is None else restored['summary']['cycle']
    stop_cycle = completed + cycles
    summaries = []
    history = ModelHistory()

    def infer(net, boards, scores):
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            b,s = dev.transfer((boards,scores))
            pi,q = net(b.contiguous(memory_format=torch.channels_last),s)
            # Blocking CPU copies also prevent reuse of compiled output storage
            # before the native consumer finishes with the result.
            return (pi.flatten(1)[:,indices].float().cpu().contiguous(),
                    q.flatten(1)[:,indices].float().cpu().contiguous())

    run_start = perf_counter()
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
    with Arena(library,options,pinned=dev.cuda,seed=(options['seed']+completed)%(2**64)) as arena:
        model_time = cpu_time = 0.0
        arena_step = 0
        while cycles == 0 or completed < stop_cycle:
            if arena_step == 0:
                output.emit(f"\nCycle {completed+1} / Self-play")
            arena_step += 1
            t = perf_counter()
            boards,scores = arena.inputs()
            encoding_time = perf_counter()-t
            model.eval()
            torch.compiler.cudagraph_mark_step_begin()
            dev.sync(); t = perf_counter()
            pi,q = infer(network,boards,scores)
            dev.sync(); model_ms = (perf_counter()-t)*1000
            model_time += model_ms/1000
            t = perf_counter()
            full = arena.step(pi,q)
            if not full:
                cpu_ms = (perf_counter()-t+encoding_time)*1000
                cpu_time += cpu_ms/1000
                continue
            count = arena.stats()[0]
            dropped = arena.reset()
            cpu_time += perf_counter()-t+encoding_time
            output.emit(f"  Mean over {arena_step:,} steps: CPU {cpu_time*1000/arena_step:.3f} ms | Model {model_time*1000/arena_step:.3f} ms")
            output.training_header(count, dropped)
            dev.sync(); training_start = perf_counter()
            history.remember(model)
            model.train()
            loss_sum = policy_sum = value_sum = 0.0
            timing_sum = [0.0, 0.0, 0.0]
            for start in range(0,count,batch.rows//2):
                valid_positions = arena.batch(start,batch)
                b,s,target,actions,returns = dev.transfer(batch.tensors)
                b = b.contiguous(memory_format=torch.channels_last)
                valid = (rows < 2*valid_positions).float()
                optimizer.zero_grad(set_to_none=True)
                torch.compiler.cudagraph_mark_step_begin()
                forward_start = dev.stamp()
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                    logits,values = network(b,s)
                    policy_loss,value_loss = losses(logits,values,target,actions,returns,valid)
                    loss = policy_loss+value_loss
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
            output.emit(f"  Mean over {batch_number:,} batches (ms)",
                f"  Forward {timing_sum[0]/batch_number:.3f} | Backward {timing_sum[1]/batch_number:.3f}"
                f" | Optimizer {timing_sum[2]/batch_number:.3f}",
                f"  Position-weighted loss {loss_sum/count:.4f} | Policy {policy_sum/count:.4f}"
                f" | Q {value_sum/count:.4f}")
            arena.clear()
            dev.sync(); training_time = perf_counter()-training_start
            output.strength_header(options['test_games'])
            test_start = perf_counter()
            model.eval()
            evaluations = []
            for age, weights in history.opponents():
                opponent_start = perf_counter()
                # Reuse one compiled opponent and copy into its existing tensors.
                old.load_state_dict(weights)
                with Arena(library,options,evaluation=True,seed=(options['seed']+completed+1)%(2**64),pinned=dev.cuda) as test:
                    # First half: new P1. Second half: new P0. Each network still
                    # sees test_games rows, including zero rows for finished slots.
                    old_rows, new_rows = evaluation_rows(options['test_games'])
                    while True:
                        b,s = test.inputs()
                        torch.compiler.cudagraph_mark_step_begin()
                        p0,q0 = infer(previous,b[old_rows],s[old_rows])
                        p1,q1 = infer(network,b[new_rows],s[new_rows])
                        pi = torch.empty((2*test.n,80))
                        q = torch.empty_like(pi)
                        pi[old_rows], pi[new_rows] = p0, p1
                        q[old_rows], q[new_rows] = q0, q1
                        if test.step(pi,q):
                            break
                    _,wins,draws,losses_count = test.stats()
                dev.sync()
                evaluations.append(dict(age=age,opponent_cycle=completed+1-age,
                    wins=wins,draws=draws,losses=losses_count,
                    win_rate=wins/options['test_games'],
                    score_rate=(wins+0.5*draws)/options['test_games'],
                    seconds=perf_counter()-opponent_start))
            dev.sync(); test_time = perf_counter()-test_start
            latest = evaluations[0]
            wins,draws,losses_count = (latest[k] for k in ('wins','draws','losses'))
            completed += 1
            summary = dict(cycle=completed,evaluations=evaluations,states=count,dropped_states=dropped,loss=loss_sum/count,
                           model_seconds=model_time,cpu_seconds=cpu_time,training_seconds=training_time,
                           strength_test_seconds=test_time,wins=wins,draws=draws,losses=losses_count,
                           selfplay_steps=arena_step,batches=batch_number,
                           mean_cpu_ms=cpu_time*1000/arena_step,
                           mean_model_ms=model_time*1000/arena_step,
                           mean_forward_ms=timing_sum[0]/batch_number,
                           mean_backward_ms=timing_sum[1]/batch_number,
                           mean_optimizer_ms=timing_sum[2]/batch_number,
                           policy_loss=policy_sum/count,q_loss=value_sum/count)
            summary.update(timestamp=datetime.now(timezone.utc).isoformat(),
                           elapsed_seconds=perf_counter()-run_start,
                           win_rate=latest['win_rate'], score_rate=latest['score_rate'])
            if log_dir is not None:
                with (log_dir/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(summary, allow_nan=False)+'\n')
            if cycles:
                summaries.append(summary)
            output.summary(summary)
            if checkpoint is not None:
                checkpoint(model, optimizer, options, summary)
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
    def checkpoint(model,optimizer,options,summary):
        path = save(directory,model,optimizer,options,summary)
        output.emit(f'Checkpoint saved: {path}')
    run(args.library,device=args.device,compile_model=not args.no_compile,
        checkpoint=checkpoint,resume=args.resume,log_dir=log_dir)


if __name__ == '__main__':
    main()
