import time
import torch
import torch._inductor.config as ic


from config import device, batch_size, iterations, height, width
from model import alpha_lines_net


def main():
    print(f"initializing model")
    start_time = time.perf_counter()

    model = alpha_lines_net().to(device).eval()

    model = model.to(torch.bfloat16)
    model = torch.compile(model)
    ic.freezing = True

    x = torch.zeros(batch_size, 7, height, width, device=device).to(torch.bfloat16)

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

    evals = iterations * batch_size
    print(f"batch_size={batch_size}  iterations={iterations}  device={device}")
    print(f"total {evals} evals in {dt:.4f}s")
    print(f"{evals / dt:,.0f} evals/s   ({dt / iterations * 1e3:.3f} ms/step)")