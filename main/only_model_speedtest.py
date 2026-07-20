import time
import torch
import torch._inductor.config as ic


from config import device, num_parallel_games, iterations, height, width
from model import alpha_lines_net


def main():
    print(f"initializing model")
    start_time = time.perf_counter()

    model = alpha_lines_net().to(device).eval()

    # model = model.to(torch.bfloat16)
    model = torch.compile(model)
    ic.freezing = True

    x = torch.zeros(num_parallel_games, 7, height, width, device=device)#.to(torch.bfloat16)

    print({p.dtype for p in model.parameters()}, {b.dtype for b in model.buffers()}, x.dtype)

    warmup = 5
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        
        print(f"compilation and warmup done {(time.perf_counter() - start_time):.2f}s")
        start_time = time.perf_counter()

        t0 = time.perf_counter()
        for _ in range(iterations):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        print(f"main run complete {(time.perf_counter() - start_time):.2f}s")
        start_time = time.perf_counter()

    evals = iterations * num_parallel_games
    print(f"num_parallel_games={num_parallel_games}  iterations={iterations}  device={device}")
    print(f"total {evals} evals in {dt:.4f}s")
    print(f"{evals / dt:,.0f} evals/s   ({dt / iterations * 1e3:.3f} ms/step)")

main()