"""Warm GPU-resident model training benchmark; no arena, transfers or logging in timing."""
import statistics
from time import perf_counter
import torch
from config import settings
from klent.train import losses
from models.katago import KataGoNet
from models.resnet import ResNet


def run(model_name='', batch_size=0, warmup=20, iterations=100):
    if warmup < 3 or iterations < 1:
        raise ValueError('require warmup >= 3 and iterations >= 1')
    options = settings['klent']
    name = model_name or options['model']
    size = batch_size or options['train_minibatch']
    if size < 1:
        raise ValueError('batch_size must be positive')
    torch.set_num_threads(1)
    torch.manual_seed(options['seed'])
    torch.backends.cudnn.benchmark = True
    model = {'katago': KataGoNet, 'resnet': ResNet}[name]().cuda().to(
        dtype=torch.bfloat16, memory_format=torch.channels_last).train()
    net = torch.compile(model, mode='max-autotune', dynamic=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options['lr'],
                                 weight_decay=options['weight_decay'])
    # Fixed resident synthetic inputs: same architecture, shapes, dtype and loss.
    boards = torch.randn(size,5,10,16,device='cuda',dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    scores = torch.randint(0,80,(size,2),device='cuda').to(torch.bfloat16)
    target = torch.randn(size,160,device='cuda').softmax(1).to(torch.bfloat16)
    actions = torch.randint(0,160,(size,),device='cuda')
    returns = torch.rand(size,device='cuda')*2-1
    valid = torch.ones(size,device='cuda')

    def step(events=None):
        optimizer.zero_grad(set_to_none=True)
        torch.compiler.cudagraph_mark_step_begin()
        if events: events[0].record()
        pi,q = net(boards,scores)
        pl,vl = losses(pi,q,target,actions,returns,valid)
        if events: events[1].record()
        (pl+vl).backward()
        if events: events[2].record()
        optimizer.step()
        if events: events[3].record()

    print(f'Benchmark: {name}, {size:,} perspectives, BF16, channels-last, max-autotune',flush=True)
    print(f'GPU: {torch.cuda.get_device_name()} | torch {torch.__version__}',flush=True)
    print('Warming forward, backward and optimizer; compilation excluded.',flush=True)
    for _ in range(warmup): step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    events = [[torch.cuda.Event(enable_timing=True) for _ in range(4)] for _ in range(iterations)]
    start = perf_counter()
    for row in events: step(row)
    torch.cuda.synchronize()
    wall = (perf_counter()-start)*1000/iterations
    report = {}
    print('Phase          Mean ms   Median ms     Best ms')
    for label,a,b in [('Forward+loss',0,1),('Backward',1,2),('Optimizer',2,3),('Total',0,3)]:
        times = [e[a].elapsed_time(e[b]) for e in events]
        report[label] = dict(mean=statistics.mean(times),median=statistics.median(times),best=min(times))
        print(f'{label:<14} {statistics.mean(times):9.3f} {statistics.median(times):11.3f} {min(times):11.3f}')
    print(f'Wall ms/step: {wall:.3f} | Peak allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    # Outside timing: compare objective gradients on the shared trunk. Synthetic
    # targets make this a diagnostic example, not proof of real-data balance.
    optimizer.zero_grad(set_to_none=True)
    diagnostic_size = min(size,32)
    pi,q = model(boards[:diagnostic_size],scores[:diagnostic_size])
    pl,vl = losses(pi,q,target[:diagnostic_size],actions[:diagnostic_size],
                   returns[:diagnostic_size],valid[:diagnostic_size])
    params = tuple(model.parameters())
    pg = torch.autograd.grad(pl,params,retain_graph=True,allow_unused=True)
    qg = torch.autograd.grad(vl,params,allow_unused=True)
    shared = [(p,v) for p,v in zip(pg,qg) if p is not None and v is not None]
    pn = sum(p.float().square().sum() for p,v in shared).sqrt().item()
    qn = sum(v.float().square().sum() for p,v in shared).sqrt().item()
    print(f'Synthetic shared-trunk gradient norms ({diagnostic_size} rows): policy {pn:.5f} | Q {qn:.5f}')
    print('Resident synthetic workload reference; best measured time is not a hardware peak bound.')
    return report
