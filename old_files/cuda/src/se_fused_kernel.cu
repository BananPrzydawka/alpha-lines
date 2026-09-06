// 1:1 replica of triton_poi_fused_add_expand_mul_sigmoid_silu_view_6
//
//   in_out : bf16 [2048,128,10,16], channels_last  -> physical (N,H,W,C)
//   se     : bf16 [2048,128] contiguous, pre-sigmoid
//   res    : bf16, same layout as in_out
//   in_out = silu(in_out * sigmoid(se) + res)     (in place)

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
constexpr int kC      = 128;    // filters
constexpr int kHWC    = 20480;  // H*W*C
constexpr int kVec    = 8;      // bf16 per thread (16B vector)
constexpr int kThreads = 128;   // == num_warps 4
constexpr int kXBlock = kThreads * kVec;  // 1024

__device__ __forceinline__ void unpack8(const uint4 &v, float *o) {
    const __nv_bfloat162 *p = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(p[i]);
        o[2 * i]     = f.x;
        o[2 * i + 1] = f.y;
    }
}
}  // namespace

extern "C" __global__ __launch_bounds__(kThreads) void se_fused_kernel(
    __nv_bfloat16 *__restrict__ in_out,
    const __nv_bfloat16 *__restrict__ se,
    const __nv_bfloat16 *__restrict__ res)
{
    const int x = blockIdx.x * kXBlock + threadIdx.x * kVec;
    const int c = x % kC;      // x0
    const int n = x / kHWC;    // x2

    const uint4 raw_main = *reinterpret_cast<const uint4 *>(in_out + x);

    // eviction_policy='evict_last' on the SE load
    uint4 raw_se;
    unsigned long long policy;
    const __nv_bfloat16 *se_ptr = se + n * kC + c;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    asm volatile("ld.global.L1::evict_last.L2::cache_hint.v4.b32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(raw_se.x), "=r"(raw_se.y), "=r"(raw_se.z), "=r"(raw_se.w)
                 : "l"(se_ptr), "l"(policy));

    const uint4 raw_res = *reinterpret_cast<const uint4 *>(res + x);

    float m[kVec], s[kVec], r[kVec], y[kVec];
    unpack8(raw_main, m);
    unpack8(raw_se, s);
    unpack8(raw_res, r);

#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        const float g = __fdividef(1.0f, 1.0f + __expf(0.0f - s[i]));  // tl.sigmoid
        const float t = fmaf(g, m[i], r[i]);                           // mul + residual
        y[i] = __fdividef(t, 1.0f + expf(0.0f - t));                   // silu, libdevice.exp
    }

    uint4 out;
    __nv_bfloat162 *o = reinterpret_cast<__nv_bfloat162 *>(&out);
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = __floats2bfloat162_rn(y[2 * i], y[2 * i + 1]);
    *reinterpret_cast<uint4 *>(in_out + x) = out;
}

void se_fused_launch(__nv_bfloat16 *in_out, const __nv_bfloat16 *se,
                     const __nv_bfloat16 *res, long long numel, cudaStream_t stream)
{
    se_fused_kernel<<<(int)(numel / kXBlock), kThreads, 0, stream>>>(in_out, se, res);
}
