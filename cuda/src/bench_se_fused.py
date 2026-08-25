import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from se_fused import se_fused  # noqa: F401  (registers alpha::se_fused)

N, C, H, W = 2048, 128, 10, 16
NUMEL = N * C * H * W


@triton.jit
def triton_ref(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x3 = xindex
    x0 = xindex % 128
    x2 = xindex // 20480
    tmp0 = tl.load(in_out_ptr0 + x3, None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0 + 128 * x2), None, eviction_policy="evict_last").to(tl.float32)
    tmp4 = tl.load(in_ptr1 + x3, None).to(tl.float32)
    tmp5 = tmp0 * tl.sigmoid(tmp1) + tmp4
    tmp12 = (tmp5 / (libdevice.exp(-tmp5) + 1.0)).to(tl.float32)
    tl.store(in_out_ptr0 + x3, tmp12, None)


def make():
    main = torch.randn(N, C, H, W, device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last
    )
    res = torch.randn_like(main)
    se = torch.randn(N, C, device="cuda", dtype=torch.bfloat16)
    return main, se, res


def reference(main, se, res):
    t = main.float() * torch.sigmoid(se.float()).view(N, C, 1, 1) + res.float()
    return torch.nn.functional.silu(t).to(torch.bfloat16)


main, se, res = make()
gold = reference(main, se, res)

a = main.clone()
torch.ops.alpha.se_fused(a, se, res)

b = main.clone()
triton_ref[(NUMEL // 1024,)](b, se, res, NUMEL, XBLOCK=1024, num_warps=4)

print("cuda vs eager  max abs diff:", (a.float() - gold.float()).abs().max().item())
print("cuda vs triton bitwise equal:", torch.equal(a, b))

t_cuda = triton.testing.do_bench(lambda: torch.ops.alpha.se_fused(main, se, res))
t_trit = triton.testing.do_bench(
    lambda: triton_ref[(NUMEL // 1024,)](main, se, res, NUMEL, XBLOCK=1024, num_warps=4)
)
bytes_moved = 3 * 2 * NUMEL + 2 * N * C
for name, t in (("cuda", t_cuda), ("triton", t_trit)):
    print(f"{name:6s} {t * 1e3:7.2f} us   {bytes_moved / (t * 1e-3) / 1e12:5.2f} TB/s")
