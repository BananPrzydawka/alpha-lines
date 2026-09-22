"""Compact plain-text reports that also remain readable in Modal logs."""


def emit(*lines):
    print('\n'.join(lines), flush=True)


def setup(options, device, compiled):
    emit('', 'KLENT  |  '+options['model']+'  |  FP32 weights + BF16 autocast  |  '+device,
         '  Compile    '+('max-autotune' if compiled else 'disabled'),
         f"  Arena      {options['n']:,} games   |   Buffer {options['m']:,} positions",
         f"  Minibatch  {options['train_minibatch']:,} perspectives   |   One epoch",
         f"  Test       {options['test_games']:,} games   |   Seed {options['seed']}",
         f"  Alpha      {options['alpha']:g}   |   Beta {options['beta']:g}   |   Lambda {options['lambda']:.6f}"
         f"   |   Score lambda {options['discounted_score_lambda']:.6f}",
         f"  Optimizer  {options['optimizer']}   |   LR {options['lr']:g}   |   Decay {options['weight_decay']:g}")
    if compiled:
        emit('  Timings include compilation/autotuning on first use.')


def summary(row):
    loss_lines = []
    for label, loss_key, weight_key in (
        ('Policy', 'policy_loss', 'policy_loss_weight'),
        ('Q', 'q_loss', 'q_loss_weight'),
        ('Mark', 'mark_class_loss', 'mark_class_loss_weight'),
        ('Score head', 'immediate_score_loss', 'immediate_score_weight'),
        ('Future score', 'discounted_score_loss', 'discounted_score_weight'),
    ):
        raw, weight = row[loss_key], row[weight_key]
        loss_lines.append(f'    {label:<11}{raw:>10.4f} × {weight} = {raw*weight:.4f}')
    emit('', f"Cycle {row['cycle']} complete", '-'*43,
         f"  Positions     {row['states']:>12,}",
         f"  Dropped       {row['dropped_states']:>12,}",
         f"  Mean loss     {row['loss']:>12.4f}",
         *loss_lines, '')
    for label, key in [('Model','model_seconds'), ('CPU','cpu_seconds'),
                       ('Shuffle','shuffle_seconds'),
                       ('Scoring head processing','scoring_head_processing_seconds'),
                       ('Training','training_seconds'), ('Strength test','strength_test_seconds')]:
        emit(f'  {label:<24}{row[key]:>12.2f} s')
    emit('', '  Opponent       W     D     L    Score')
    for result in row['evaluations']:
        w,d,l = (result[k] for k in ('wins','draws','losses'))
        score = (w+0.5*d)/(w+d+l)
        emit(f"  {result['age']:>2} cycles ago {w:5d} {d:5d} {l:5d}  {score:6.1%}")


def timing_footer(checkpoint, metrics, reporting, other, total):
    emit('', f'  {"Checkpoint":<24}{checkpoint:>12.2f} s',
         f'  {"Metrics":<24}{metrics:>12.2f} s',
         f'  {"Reporting":<24}{reporting:>12.2f} s',
         f'  {"Other":<24}{other:>12.2f} s',
         f'  {"Cycle total":<24}{total:>12.2f} s')
    emit('-'*43)
