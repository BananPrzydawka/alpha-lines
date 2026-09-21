"""Compact plain-text reports that also remain readable in Modal logs."""


def emit(*lines):
    print('\n'.join(lines), flush=True)


def setup(options, device, compiled):
    emit('', 'KLENT  |  '+options['model']+'  |  BF16  |  '+device,
         '  Compile    '+('max-autotune' if compiled else 'disabled'),
         f"  Arena      {options['n']:,} games   |   Buffer {options['m']:,} positions",
         f"  Minibatch  {options['train_minibatch']:,} perspectives   |   One epoch",
         f"  Test       {options['test_games']:,} games   |   Seed {options['seed']}",
         f"  Alpha      {options['alpha']:g}   |   Beta {options['beta']:g}   |   Lambda {options['lambda']:.6f}",
         f"  Optimizer  {options['optimizer']}   |   LR {options['lr']:g}   |   Decay {options['weight_decay']:g}")
    if compiled:
        emit('  Timings include compilation/autotuning on first use.')


def training_header(count, dropped):
    emit('', f'Training / {count:,} positions / {dropped:,} unfinished positions dropped')


def strength_header(games):
    emit('', f'Strength test / {games:,} games per opponent / equal games on each side / policy sampling')


def summary(row):
    emit('', f"Cycle {row['cycle']} complete", '-'*43,
         f"  Positions     {row['states']:>12,}",
         f"  Dropped       {row['dropped_states']:>12,}",
         f"  Mean loss     {row['loss']:>12.4f}", '')
    for label, key in [('Model','model_seconds'), ('CPU','cpu_seconds'),
                       ('Training','training_seconds'), ('Strength test','strength_test_seconds')]:
        emit(f'  {label:<14}{row[key]:>12.2f} s')
    emit('', '  Opponent       W     D     L    Score')
    for result in row['evaluations']:
        w,d,l = (result[k] for k in ('wins','draws','losses'))
        score = (w+0.5*d)/(w+d+l)
        emit(f"  {result['age']:>2} cycles ago {w:5d} {d:5d} {l:5d}  {score:6.1%}")
    emit('-'*43)
