import torch
import torch.cuda.profiler as prof
import torch._inductor.config as cfg

from config import device, height, width, num_parallel_games
from model import alpha_lines_net


# cfg.max_autotune = True
# cfg.max_autotune_gemm_backends = "TRITON"    # ATEN,TRITON
# cfg.max_autotune_conv_backends = "TRITON"
# cfg.force_disable_caches = True

cfg.trace.enabled = True


def main():

    model = alpha_lines_net().to(device).eval().to(torch.bfloat16)
    model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model, mode="max-autotune")
    
    x = torch.zeros(num_parallel_games, 7, height, width, device=device, dtype=torch.bfloat16)
    x = x.to(memory_format=torch.channels_last)

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