"""Compact plain-text reports that also remain readable in Modal logs."""


def emit(*lines):
    print('\n'.join(lines), flush=True)


def setup(options, device, compiled):
    emit('', 'KLENT  |  '+options['model']+'  |  FP32 weights + BF16 autocast  |  '+device,
         '  Compile    '+('default' if compiled else 'disabled'),
         f"  Arena      {options['n']:,} games   |   Buffer {options['m']:,} positions",
         f"  Minibatch  {options['train_minibatch']:,} perspectives   |   One epoch",
         f"  Gradient accumulation {'enabled' if options['gradient_accumulation'] else 'disabled'}",
         f"  Test       {options['test_games']:,} games   |   Seed {options['seed']}",
         f"  Anchors    score > {options['anchor_thresholds'][0]:.0%},"
         f" {options['anchor_thresholds'][1]:.0%}, {options['anchor_thresholds'][2]:.0%}",
         f"  Policy target {'minibatch recalculation' if options['policy_recalculation'] else 'stored self-play'}",
         f"  Reference  {options['reference_checkpoint']}",
         f"  Alpha      {options['alpha']:g}   |   Beta {options['beta']:g}   |   Lambda {options['lambda']:.6f}",
         f"  Self-play uniform exploration {options['exploration_fraction']:.1%}",
         f"  Optimizer  {options['optimizer']}   |   LR {options['lr']:g}   |   Decay {options['weight_decay']:g}")
    if compiled:
        emit('  Timings include compilation on first use.')


def summary(row, timing):
    loss_lines = []
    for label, loss_key, weight_key in (
        ('Policy', 'policy_loss', 'policy_loss_weight'),
        ('Action value', 'q_loss', 'q_loss_weight'),
    ):
        raw, weight = row[loss_key], row[weight_key]
        loss_lines.append(f'    {label:<16}{raw:>8.4f} × {weight} = {raw*weight:.4f}')
    lines = ['', f"Cycle {row['cycle']} complete", '-'*43,
         f"  Positions     {row['states']:>12,}",
         f"  Dropped       {row['dropped_states']:>12,}",
         f"  Mean loss     {row['loss']:>12.4f}",
         *loss_lines, '']
    for label, key in [('Total','total'), ('CPU','cpu'),
                       ('Model self play','selfplay_model'),
                       ('Model training','training_model'),
                       ('Strength test','strength_test'), ('Other','other')]:
        lines.append(f'  {label:<24}{timing[key]:>12.2f} s')
    lines.extend(('', '  Opponent                  W     D     L    Score'))
    for result in row['evaluations']:
        w,d,l = (result[k] for k in ('wins','draws','losses'))
        score = (w+0.5*d)/(w+d+l)
        opponent = ('Previous' if result['kind'] == 'previous'
                    else f"Anchor {result['anchor_index']} (cycle {result['opponent_cycle']})"
                    if result['kind'] == 'anchor'
                    else f"Checkpoint {result['opponent_cycle']}")
        lines.append(f"  {opponent:<24}{w:5d} {d:5d} {l:5d}  {score:6.1%}")
    for update in row['anchor_updates']:
        lines.append(f"  Anchor {update['anchor']} advanced: cycle {update['old_cycle']}"
                     f" → {update['new_cycle']} at {update['score_rate']:.1%}")
    lines.append('-'*43)
    emit(*lines)
