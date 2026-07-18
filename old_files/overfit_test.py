import time
import torch
import torch.nn.functional as F

from game import batched_lines_game
from monte_carlo import BatchedMCTS
from config import height, width, board_size, device, iterations, learning_rate, weight_decay, log_interval
from model import alpha_lines_net


def main(prof=None):
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    game = batched_lines_game.import_print("""
+--------------------------------+
|oo  oo  oo  oo  ||  ||  ||  ||  |
|  oo  oo  oo  oo  ||  ||  ||  |||
|oo  oo  oo  oo  ||  ||  ||  ||  |
|  oo  oo  oo  oo  ||  ||  ||  |||
|oo  oo  oo  oo  ||  ||  ||  ||  |
|  oo  oo  oo  oo  ||  ||  ||  |||
|oo  oo  oo  oo  ||  ||  ||  ||  |
|  oo  oo  oo  --  ||  ||  ||  |||
|oo  oo  oo  oo  --  ||  ||  ||  |
|  oo  oo  oo  --  ||  ||  ||  |||
+--------------------------------+
""", device=device)
    
    game.print_state(0)

    model = alpha_lines_net().to(device)
    model.eval()

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    enc_p0 = game.get_encoded_states(player=0).detach()  # (1, C, H, W)
    enc_p1 = game.get_encoded_states(player=1).detach()
    enc = torch.cat([enc_p0, enc_p1], dim=0)             # (2, C, H, W), fixed

    for step in range(1, iterations + 1):
        t0 = time.time()

        mcts = BatchedMCTS()
        with torch.no_grad():
            # Unpack the counts (pi) and values (q) from the MCTS search
            pi0, pi1, q0, q1 = mcts.search(game, model)

        tgt_p0 = pi0.view(1, height, width).detach()
        tgt_p1 = pi1.view(1, height, width).detach()
        target = torch.cat([tgt_p0.reshape(1, board_size),
                            tgt_p1.reshape(1, board_size)], dim=0)  # (2, board_size)

        # Flatten Q values for easier indexing alongside target probabilities
        q_values = torch.cat([q0.reshape(1, board_size),
                              q1.reshape(1, board_size)], dim=0)

        opt.zero_grad(set_to_none=True)
        logits, _, _, _ = model(enc)
        logp = F.log_softmax(logits.flatten(1), dim=1)
        loss = -(target * logp).sum(dim=1).mean()
        loss.backward()
        opt.step()

        dt_ms = (time.time() - t0) * 1000

        if step % log_interval == 0 or step == 1:
            with torch.no_grad():
                kl_per = (target * (target.clamp_min(1e-12).log() - logp)).sum(dim=1)
                kl = kl_per.mean().item()
                kl_worst = kl_per.max().item()

            p0_legal = torch.nonzero(tgt_p0.flatten() > 1e-6, as_tuple=False).flatten().tolist()
            p1_legal = torch.nonzero(tgt_p1.flatten() > 1e-6, as_tuple=False).flatten().tolist()

            rows = [
                ("enc_p0", 0, p0_legal),
                ("enc_p1", 1, p1_legal),
            ]

            print(f"\nstep {step:4d} | {dt_ms:6.1f} ms | CE {loss.item():.5f} | KL {kl:.3e} (worst {kl_worst:.3e})")
            for label, row_idx, legal_cells in rows:
                t = target[row_idx]
                q = q_values[row_idx]
                print(f"\n  {label}:")
                print(f"    {'(r,c)':>7} | {'tgt prob':>8} | {'value (Q)':>9}")
                for idx in legal_cells:
                    r, c = divmod(idx, width)
                    print(f"    ({r},{c:2d})  | {t[idx].item():>8.4f} | {q[idx].item():>9.4f}")

        if prof is not None:
            prof.step()