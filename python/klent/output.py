"""Compact plain-text reports that also remain readable in Modal logs."""
from pathlib import Path


def emit(*lines):
    print('\n'.join(lines), flush=True)


def setup(options, device, compiled):
    one_side = options['fixed_opponent'] and not options['both_sides']
    minibatch_rows = options['train_minibatch']//2 if one_side else options['train_minibatch']
    emit('', 'KLENT  |  '+options['model']+'  |  FP32 weights + BF16 autocast  |  '+device,
         '  Compile    '+(('max-autotune' if options.get('compile_max_autotune', False)
                           else 'default') if compiled else 'disabled'),
         f"  Arena      {options['n']:,} games   |   Buffer {options['m']:,} positions",
         f"  Minibatch  {minibatch_rows:,} perspectives "
         f"({1 if one_side else 2} per position)   |   One epoch",
         f"  Test       {options['test_games']:,} games   |   Seed {options['seed']}",
         f"  Anchors    every {options['anchor_interval']} cycles, starting at cycle 0",
         f"  Test       every {options['test_interval']} cycles",
         f"  Training   {'fixed opponent' if options['fixed_opponent'] else 'self play'}",
         *((f"  Opponent   {Path(options['opponent_checkpoint']).name}",)
           if options.get('opponent_checkpoint') else ()),
         f"  References {len(options['reference_checkpoints'])}",
         f"  Alpha      {options['alpha']:g}   |   Beta {options['beta']:g}   |   Lambda {options['lambda']:.6f}",
         f"  Self-play uniform exploration {options['exploration_fraction']:.1%}",
         f"  Optimizer  {options['optimizer']}   |   LR {options['lr']:g}   |   Decay {options['weight_decay']:g}")
    if compiled:
        emit('  Timings include compilation on first use.')


def summary(row, timing):
    lines = ['', f"Cycle {row['cycle']} complete", '-'*43,
         f"  Positions     {row['states']:>12,}",
         f"  Dropped       {row['dropped_states']:>12,}",
         f"  Mean loss     {row['loss']:>12.4f}",
         f"    Policy          {row['policy_loss']:>8.4f}",
         f"    Action value    {row['q_loss']:>8.4f}", '']
    if row['training_opponent_cycle'] is not None:
        lines.insert(3, f"  Training opponent     cycle {row['training_opponent_cycle']}")
    for label, key in [('Total','total'), ('CPU','cpu'),
                       ('Model self play','selfplay_model'),
                       ('Model training','training_model'),
                       ('Strength test','strength_test'), ('Other','other')]:
        lines.append(f'  {label:<24}{timing[key]:>12.2f} s')
    if row['evaluations']:
        def label(result):
            if result['kind'] == 'previous':
                return 'Previous'
            if result['kind'] == 'anchor':
                return f"Anchor cycle {result['opponent_cycle']}"
            path = Path(result['reference_path'])
            name = path.name
            if path.parent.parent.name == 'imports':
                index, separator, original = name.partition('-')
                if separator and index.isdecimal():
                    return original
            return name

        width = max(24, *(len(label(result)) for result in row['evaluations']))
        lines.extend(('', f"  {'Opponent':<{width}}{'W':>5} {'D':>5} {'L':>5}  {'Score':>6}"))
        for result in row['evaluations']:
            w,d,l = (result[k] for k in ('wins','draws','losses'))
            score = (w+0.5*d)/(w+d+l)
            lines.append(f"  {label(result):<{width}}{w:5d} {d:5d} {l:5d}  {score:6.1%}")
    else:
        lines.extend(('', '  Strength test skipped'))
    for update in row['anchor_updates']:
        lines.append(f"  Anchor saved: cycle {update['new_cycle']}")
    lines.append('-'*43)
    emit(*lines)
