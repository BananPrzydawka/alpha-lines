"""Balanced checkpoint round robin on Modal.

Run from the project root with repeated --oponent MODEL_PATH arguments:
    PYTHONPATH=python uv run modal run -m modal_matchups --oponent a.pt --oponent b.pt
"""

import modal

from config import resources
from modal_image import base_image, project_dir, volume, with_project_files


app = modal.App('alphalines-matchups')
compile_cache = modal.Volume.from_name('alphalines-klent-compile-cache', create_if_missing=True)
matchup_image = (
    with_project_files(
        base_image.apt_install('build-essential')
        .run_commands('curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh',
                      'sh /tmp/rustup.sh -y --profile minimal')
        .env({'TORCHINDUCTOR_CACHE_DIR': '/compile-cache/inductor',
              'TRITON_CACHE_DIR': '/compile-cache/triton',
              'CUDA_CACHE_PATH': '/compile-cache/cuda'})
    ).add_local_dir(project_dir/'rust', remote_path='/root/rust', ignore=['target/'])
)


def validate_matchups(options, paths):
    if len(paths) < 2:
        raise ValueError('Pass at least two --oponent checkpoints')
    if type(options['games']) is not int or options['games'] < 2 or options['games'] % 2:
        raise ValueError('matchups.games must be a positive even integer')
    if type(options['seed']) is not int or not 0 <= options['seed'] < 2**64:
        raise ValueError('matchups.seed must fit u64')
    for key in ('compile_model', 'compile_max_autotune'):
        if type(options[key]) is not bool:
            raise ValueError(f'matchups.{key} must be a boolean')


def play_pair(library, first, second, games, seed, device):
    """Return W/D/L from the first model's perspective."""
    import torch
    from config import settings
    from klent.native import Arena
    from klent.train import evaluation_rows

    options = dict(settings['klent'], test_games=games)
    first_rows, second_rows = evaluation_rows(games)
    squares = torch.arange(80, device=device)
    board_indices = squares//8*16 + 2*(squares%8) + (squares//8%2)
    use_cuda = torch.device(device).type == 'cuda'

    def infer(model, boards):
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            inputs = boards.to(device, non_blocking=use_cuda)
            policy, _ = model(inputs.contiguous(memory_format=torch.channels_last))
            return policy.flatten(1)[:, board_indices].float().cpu().contiguous()

    with Arena(library, options, evaluation=True, seed=seed, pinned=use_cuda) as arena:
        while True:
            boards = arena.inputs()
            torch.compiler.cudagraph_mark_step_begin()
            first_policy = infer(first, boards[first_rows])   # games rows in one call
            second_policy = infer(second, boards[second_rows]) # games rows in one call
            policy = torch.empty((2*games, 80))
            policy[first_rows] = first_policy
            policy[second_rows] = second_policy
            if arena.step(policy, torch.zeros_like(policy)):
                break
        # Native evaluation records the second model as its "new" player.
        second_wins, draws, first_wins = arena.stats()[1:]
        return first_wins, draws, second_wins


def run_matchups(library, paths, labels, options, device='cuda'):
    import torch
    from klent.train import load_reference_model

    validate_matchups(options, paths)
    models = []
    compile_options = {'dynamic': False}
    if options['compile_max_autotune']:
        compile_options['mode'] = 'max-autotune'
    for path in paths:
        model, cycle = load_reference_model(path, device)
        if options['compile_model']:
            model = torch.compile(model, **compile_options)
        models.append((model, cycle))

    print(f"{len(models)} checkpoints | {len(models)*(len(models)-1)//2} matchups | "
          f"{options['games']} games per matchup", flush=True)
    for index, ((_, cycle), label) in enumerate(zip(models, labels), 1):
        print(f'  {index:>2}. {label} (cycle {cycle})', flush=True)
    results = {}
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            wins, draws, losses = play_pair(library, models[i][0], models[j][0],
                                            options['games'], options['seed'], device)
            results[i, j] = wins, draws, losses
            score = (wins + draws/2) / options['games']
            print(f'  {labels[i]} vs {labels[j]}: {wins} W / {draws} D / {losses} L '
                  f'({score:.1%} for {labels[i]})', flush=True)

    width = max(8, max(map(len, labels))+2)
    print('\nScore matrix: row model versus column model (W + ½D)', flush=True)
    print(f"{'':>{width}}" + ''.join(f'{label:>{width}}' for label in labels), flush=True)
    for i in range(len(models)):
        cells = []
        for j in range(len(models)):
            if j == i:
                cells.append('—')
            elif i < j:
                wins, draws, _ = results[i, j]
                cells.append(f'{(wins+draws/2)/options["games"]:.1%}')
            else:
                _, draws, losses = results[j, i]
                cells.append(f'{(losses+draws/2)/options["games"]:.1%}')
        print(f'{labels[i]:>{width}}' + ''.join(f'{cell:>{width}}' for cell in cells), flush=True)
    return results


@app.function(image=matchup_image, gpu=resources['gpu'], timeout=resources['timeout'],
              volumes={'/checkpoints': volume.with_mount_options(read_only=True),
                       '/compile-cache': compile_cache})
def compare(paths: tuple[str, ...], labels: tuple[str, ...]):
    import subprocess
    from config import settings

    subprocess.run(['/root/.cargo/bin/cargo', 'build', '--release', '--locked',
                    '--manifest-path', '/root/rust/Cargo.toml'], check=True)
    options = settings['matchups']
    try:
        return run_matchups('/root/rust/target/release/libalpha_lines_game.so',
                            paths, labels, options)
    finally:
        if options['compile_model']:
            compile_cache.commit()


@app.local_entrypoint()
def main(*args: str):
    import argparse
    from pathlib import Path
    from uuid import uuid4
    from config import settings

    parser = argparse.ArgumentParser(description='Balanced checkpoint matchup matrix on Modal')
    parser.add_argument('--oponent', '--opponent', dest='opponents', action='append',
                        required=True, metavar='MODEL_PATH',
                        help='Checkpoint to include; repeat for each model')
    parsed = parser.parse_args(args)
    if len(parsed.opponents) < 2:
        parser.error('Pass at least two --oponent checkpoints')
    validate_matchups(settings['matchups'], parsed.opponents)
    local_paths = {path: Path(path) for path in parsed.opponents
                   if not path.startswith('/checkpoints/')}
    for path, local in local_paths.items():
        if not local.is_file():
            parser.error(f'Checkpoint does not exist locally or on /checkpoints: {path}')

    staged = {}
    prefix = None
    if local_paths:
        prefix = f'imports/matchups/{uuid4().hex}'
        with volume.batch_upload() as upload:
            for index, (path, local) in enumerate(local_paths.items()):
                remote = f'{prefix}/{index}-{local.name}'
                upload.put_file(str(local), remote)
                staged[path] = f'/checkpoints/{remote}'
    try:
        compare.remote(tuple(staged.get(path, path) for path in parsed.opponents),
                       tuple(Path(path).name.removesuffix('.pt') for path in parsed.opponents))
    finally:
        if prefix is not None:
            volume.remove_file(prefix, recursive=True)
