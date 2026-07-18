import io
import sys
import torch
from config import height, width
from game import batched_lines_game


def capture_print(game: batched_lines_game, index: int) -> str:
    buf = io.StringIO()
    sys.stdout, old = buf, sys.stdout
    game.print_state(index=index, player=0)
    sys.stdout = old
    return buf.getvalue()


def collect_game_at_count(target_count: int, device: str) -> str:
    """Run single games, restarting whenever a collision causes the legal move
    count to jump past target_count. Returns print_state text at exactly target_count."""
    uniform = torch.ones((1, height, width), device=device)
    while True:
        game = batched_lines_game(num_games=1, device=device)
        while not game.finished[0]:
            _, _, count_p0, _ = game.get_legal_masks()
            count = int(count_p0[0].item())
            if count == target_count:
                return capture_print(game, 0)
            if count < target_count:
                break  # jumped past target due to collision; restart
            game.distribution_step(uniform, uniform)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    targets = [3] * 64 + [4] * 64
    captured = []

    for i, target in enumerate(targets):
        text = collect_game_at_count(target, device)
        captured.append(text)
        print(f"Collected game {i+1}/128 (target={target})", file=sys.stderr)

    # --- Individual prints ---
    print("=" * 60)
    print("INDIVIDUAL GAME STATES")
    print("=" * 60)
    for i, text in enumerate(captured):
        print(f"# Game {i:3d} | legal_moves_remaining={targets[i]}")
        print(text)

    # --- Batch-importable block ---
    print("=" * 60)
    print("BATCH FORMAT — paste into batched_lines_game.import_prints()")
    print("=" * 60)
    print("".join(captured))