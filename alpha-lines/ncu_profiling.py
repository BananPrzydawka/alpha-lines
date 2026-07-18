import torch
import torch.cuda.profiler as prof

from config import device, height, width, num_parallel_games
from model import alpha_lines_net

def main():
    model = alpha_lines_net().to(device).eval().to(torch.bfloat16)
    x = torch.zeros(num_parallel_games, 7, height, width, device=device, dtype=torch.bfloat16)

    with torch.no_grad():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()

        prof.start()
        model(x)
        torch.cuda.synchronize()
        prof.stop()

if __name__ == "__main__":
    main()