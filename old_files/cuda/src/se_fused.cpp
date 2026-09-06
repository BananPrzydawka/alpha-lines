#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

void se_fused_launch(__nv_bfloat16 *, const __nv_bfloat16 *, const __nv_bfloat16 *,
                     long long, cudaStream_t);

void se_fused(at::Tensor x, at::Tensor se, at::Tensor res) {
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && se.scalar_type() == at::kBFloat16 &&
                res.scalar_type() == at::kBFloat16, "bf16 only");
    TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), "x must be channels_last");
    TORCH_CHECK(res.is_contiguous(at::MemoryFormat::ChannelsLast), "res must be channels_last");
    TORCH_CHECK(se.is_contiguous() && se.dim() == 2, "se must be contiguous (N, C)");
    TORCH_CHECK(x.size(1) == 128 && x.size(2) * x.size(3) * 128 == 20480, "shape hardcoded");
    TORCH_CHECK(x.numel() % 1024 == 0 && x.sizes() == res.sizes());

    se_fused_launch(reinterpret_cast<__nv_bfloat16 *>(x.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16 *>(se.data_ptr()),
                    reinterpret_cast<const __nv_bfloat16 *>(res.data_ptr()),
                    x.numel(), at::cuda::getCurrentCUDAStream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("se_fused", &se_fused); }
