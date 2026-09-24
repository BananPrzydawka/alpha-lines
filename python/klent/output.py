"""Compact plain-text reports that also remain readable in Modal logs."""


def emit(*lines):
    print('\n'.join(lines), flush=True)


def setup(options, device, compiled):
    emit('', 'KLENT  |  '+options['model']+'  |  FP32 weights + BF16 autocast  |  '+device,
         '  Compile    '+('default' if compiled else 'disabled'),
         f"  Arena      {options['n']:,} games   |   Buffer {options['m']:,} positions",
         f"  Minibatch  {options['train_minibatch']:,} perspectives   |   One epoch",
         f"  Test       {options['test_games']:,} games   |   Seed {options['seed']}",
         f"  Alpha      {options['alpha']:g}   |   Beta {options['beta']:g}   |   Lambda {options['lambda']:.6f}"
         f"   |   Score lambda {options['discounted_score_lambda']:.6f}",
         f"  Self-play uniform exploration {options['exploration_fraction']:.1%}",
         f"  Discounted mark lambda {options['discounted_mark_lambda']:.6f}",
         f"  Optimizer  {options['optimizer']}   |   LR {options['lr']:g}   |   Decay {options['weight_decay']:g}")
    if compiled:
        emit('  Timings include compilation on first use.')


def summary(row, timing):
    loss_lines = []
    for label, loss_key, weight_key in (
        ('Policy', 'policy_loss', 'policy_loss_weight'),
        ('Opponent policy', 'opponent_policy_loss', 'opponent_policy_weight'),
        ('Action value', 'q_loss', 'q_loss_weight'),
        ('Mark', 'mark_class_loss', 'mark_class_loss_weight'),
        ('Future mark', 'discounted_mark_loss', 'discounted_mark_weight'),
        ('Score', 'immediate_score_loss', 'immediate_score_weight'),
        ('Future score', 'discounted_score_loss', 'discounted_score_weight'),
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
        opponent = (f"Previous / anchor {result['opponent_cycle']}"
                    if result['kind'] == 'previous' and result['is_anchor']
                    else 'Previous' if result['kind'] == 'previous'
                    else f"Anchor cycle {result['opponent_cycle']}")
        lines.append(f"  {opponent:<24}{w:5d} {d:5d} {l:5d}  {score:6.1%}")
    lines.append('-'*43)
    emit(*lines)
