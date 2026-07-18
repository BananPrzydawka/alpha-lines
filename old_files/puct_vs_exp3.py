import time
import torch
import torch.nn as nn
from game import batched_lines_game
from mcts_puct import PuctMCTS
from mcts_exp3 import Exp3MCTS
from config import height, width, board_size, device, mcts_num_simulations

SIM_COUNTS = [mcts_num_simulations]


class UniformModel(nn.Module):
    """Uniform policy, zero value — isolates MCTS selection behaviour."""
    def forward(self, x, apply_softmax=False):
        b = x.shape[0]
        policy = torch.full((b, board_size), 1 / board_size, device=x.device)
        value  = torch.zeros(b, 3, device=x.device)
        value[:, 1] = 1.0  # all mass on draw → net value = 0
        return policy, value, None, None


def print_results(label, pi0, pi1, q0, q1, legal, elapsed_ms):
    print(f"\n  [{label}]  {elapsed_ms:.0f} ms")
    for tag, pi, q, sign in [("p0", pi0, q0, 1), ("p1", pi1, q1, -1)]:
        print(f"    enc_{tag}:  (r,c) | tgt prob | Q")
        for idx in legal:
            r, c = divmod(idx, width)
            print(f"      ({r},{c:2d})  | {pi[idx].item():.4f}")


def main():
    torch.set_float32_matmul_precision("high")

    game = batched_lines_game.import_print("""
+--------------------------------+
|||  ||  oo  ||  oo  oo  ||  ||  |
|  oo  ||  ||  ||  ||  oo  ||  --|
|oo  oo  ||  oo  oo  ||  oo  ||  |
|  ##  ##  ||  oo  ||  ||  oo  oo|
|oo  ##  oo  oo  ||  oo  oo  --  |
|  ##  ##  oo  ||  ||  ||  ||  |||
|oo  oo  ||  oo  oo  oo  ||  oo  |
|  oo  ||  oo  ||  oo  oo  ||  |||
|||  ||  oo  ||  oo  ||  oo  oo  |
|  ||  ||  ||  --  ||  oo  ||  oo|
+--------------------------------+

""", device=device)

    model = UniformModel().to(device)
    model.eval()

    playable_val = game.PLAYABLE_SQUARE
    legal = torch.nonzero(game.boards[0].flatten() == playable_val, as_tuple=False).flatten().tolist()

    for n in SIM_COUNTS:
        print(f"\n{'='*64}")
        print(f"  {n:,} simulations")
        print(f"{'='*64}")

        mcts = PuctMCTS()
        mcts.num_sims = n
        with torch.no_grad():
            t0 = time.time()
            pi0, pi1, q0, q1 = mcts.search(game, model)
            dt = (time.time() - t0) * 1000
        print_results("PUCT / PuctMCTS", pi0, pi1, q0, q1, legal, dt)

        # rmcts = Exp3MCTS(num_sims=n)
        # with torch.no_grad():
        #     t0 = time.time()
        #     pi0, pi1, q0, q1 = rmcts.search(game, model)
        #     dt = (time.time() - t0) * 1000
        # print_results("Regret / Exp3MCTS", pi0, pi1, q0, q1, legal, dt)