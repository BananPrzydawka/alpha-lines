import re
import sys
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# from game_torch import batched_lines_game as GameTorch
from game import batched_lines_game as GameNumpy
# from mcts_puct_torch import BatchedPuctMCTS as BatchedPuctMCTSTorch
from mcts_puct import BatchedPuctMCTS
from mcts_exp3 import BatchedExp3MCTS
from config import board_size, device, num_parallel_games, mcts_num_simulations

GAMES_FILE = Path(__file__).parent / "_testing_games.txt"
BORDER_RE = re.compile(r'^\+-+\+$')


class UniformModel(nn.Module):
    def forward(self, x, apply_softmax=False):
        b = x.shape[0]
        policy = torch.full((b, board_size), 1.0 / board_size, device=x.device)
        value = torch.zeros(b, 3, device=x.device)
        value[:, 1] = 1.0
        return policy, value, None, None


def parse_boards(text: str) -> list[str]:
    """
    Line-based parser. A board is delimited by two border lines
    (e.g. '+--------------------------------+'); everything between them
    (inclusive) is captured as one block. Lines outside border pairs
    (e.g. 'Game Index: ...') are ignored.
    """
    boards = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if BORDER_RE.match(lines[i].strip()):
            block = [lines[i]]
            i += 1
            while i < len(lines) and not BORDER_RE.match(lines[i].strip()):
                block.append(lines[i])
                i += 1
            if i < len(lines):
                block.append(lines[i])
                i += 1
            boards.append("\n".join(block))
        else:
            i += 1
    return boards


def legal_values(policy, mask) -> list[float]:
    """policy, mask: torch tensor or numpy array, either accepted via np.asarray."""
    policy = np.asarray(policy)
    idx = np.nonzero(np.asarray(mask))[0]
    return [round(float(policy[i]), 4) for i in idx]


def main(
    use_real_model: bool = False,
    n_sims: int = mcts_num_simulations,
    limit: int | None = None,
):
    real_start = time.perf_counter()

    torch.set_float32_matmul_precision("high")

    raw_boards = GAMES_FILE.read_text()
    boards = parse_boards(raw_boards)
    if not boards:
        sys.exit(f"No board blocks found in {GAMES_FILE}")
    if limit:
        boards = boards[:limit]
        raw_boards = "\n".join(boards)

    n_boards = len(boards)
    n_chunks = math.ceil(n_boards / num_parallel_games)

    print(f"Boards        : {n_boards}  (from {GAMES_FILE.name})")
    print(f"Simulations   : {n_sims}")
    print(f"Parallel games: {num_parallel_games}  ({n_chunks} batched call(s))")
    print(f"Device        : {device}")
    print(f"Model         : {'alpha_lines_net' if use_real_model else 'UniformModel'}")
    print()

    if use_real_model:
        from model import alpha_lines_net
        model = alpha_lines_net().to(device)
        model.eval()
    else:
        model = UniformModel().to(device)

    # torch_mcts = BatchedPuctMCTSTorch()
    numpy_mcts = BatchedExp3MCTS()

    # print("Phase 1: torch PUCT MCTS...")
    # with torch.no_grad():
    #     games_torch = GameTorch.import_prints(raw_boards)
    #     torch_results = torch_mcts.search(games_torch, model)[:num_parallel_games]
    # print("  done\n")

    print("Phase 2: numpy PUCT MCTS...")
    with torch.no_grad():
        games_numpy = GameNumpy.import_prints(raw_boards)
        numpy_results = numpy_mcts.search(games_numpy, model)[:num_parallel_games]
    print("  done\n")

    # ── Phase 3: print policy values at legal indices ────────────────────────
    # for idx, (t_res, n_res) in enumerate(zip(torch_results, numpy_results)):
    #     tpi0, tpi1, _, _ = t_res
    #     npi0, npi1, _, _ = n_res

    #     game = GameNumpy.import_prints(boards[idx])
    #     mask_p0, mask_p1, _, _ = game.get_legal_masks()
    #     mask_p0, mask_p1 = mask_p0[0], mask_p1[0]

    #     print(f"── Game {idx} " + "─" * 50)
    #     print(f"  torch  p0: {legal_values(tpi0, mask_p0)}")
    #     print(f"  numpy  p0: {legal_values(npi0, mask_p0)}")
    #     print(f"  torch  p1: {legal_values(tpi1, mask_p1)}")
    #     print(f"  numpy  p1: {legal_values(npi1, mask_p1)}")
    #     print()

    end_time = time.perf_counter()
    total_time = end_time - real_start
    print(f"real execution time: {(total_time):.2f}")
