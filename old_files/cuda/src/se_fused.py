import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.cpp_extension import load

_dir = Path(__file__).parent
_ext = load(
    name="se_fused_ext",
    sources=[str(_dir / "se_fused.cpp"), str(_dir / "se_fused_kernel.cu")],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
    verbose=False,
)


@torch.library.custom_op("alpha::se_fused", mutates_args={"x"})
def se_fused(x: torch.Tensor, se: torch.Tensor, res: torch.Tensor) -> None:
    _ext.se_fused(x, se, res)


@se_fused.register_fake
def _(x, se, res) -> None:
    return None


class fused_res_block(nn.Module):
    """res_block with sigmoid/mul/add/silu replaced by one custom kernel.

    Shares parameter objects with the source block, so a fused model built
    from a deepcopy stays weight-identical to its source."""

    def __init__(self, src):
        super().__init__()
        self.conv1, self.bn1 = src.conv1, src.bn1
        self.conv2, self.bn2 = src.conv2, src.bn2
        self.squeeze = src.se.squeeze
        self.fc1 = src.se.excite[0]
        self.act = src.se.excite[1]
        self.fc2 = src.se.excite[2]   # excite[3] is Sigmoid -> done in the kernel
        self.silu = src.silu

    def forward(self, x):
        residual = x
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        b, c, _, _ = out.shape
        y = self.fc2(self.act(self.fc1(self.squeeze(out).view(b, c))))
        torch.ops.alpha.se_fused(out, y, residual)
        return out


def fuse_model(model):
    model.tower = nn.Sequential(*[fused_res_block(b) for b in model.tower])
    return model