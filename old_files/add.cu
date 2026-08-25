// Elementwise bf16 add over the same shape as the Inductor kernel:
//   2048 x 128 x 10 x 16 = 41'943'040 elems = 80 MiB per tensor.
// Traffic per launch: 2 loads + 1 store = 240 MiB.

#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                        \
              cudaGetErrorString(e_));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

using bf16 = __nv_bfloat16;

constexpr long long N = 2048LL * 128 * 10 * 16;
constexpr long long N8 = N / 8;  // one float4 = 16B = 8 bf16
constexpr size_t BYTES = N * sizeof(bf16);

// ---------------------------------------------------------------- kernels

__global__ void add_scalar(bf16 *__restrict__ o, const bf16 *__restrict__ a,
                           const bf16 *__restrict__ b, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) o[i] = __hadd(a[i], b[i]);
}

__device__ __forceinline__ float4 add8(float4 va, float4 vb) {
  __nv_bfloat162 *pa = reinterpret_cast<__nv_bfloat162 *>(&va);
  __nv_bfloat162 *pb = reinterpret_cast<__nv_bfloat162 *>(&vb);
#pragma unroll
  for (int k = 0; k < 4; ++k) pa[k] = __hadd2(pa[k], pb[k]);
  return va;
}

__global__ void add_vec8(bf16 *__restrict__ o, const bf16 *__restrict__ a,
                         const bf16 *__restrict__ b, long long n8) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n8) return;
  reinterpret_cast<float4 *>(o)[i] =
      add8(reinterpret_cast<const float4 *>(a)[i],
           reinterpret_cast<const float4 *>(b)[i]);
}

// Persistent grid: launch exactly W waves, let each thread loop.
__global__ void add_vec8_gs(bf16 *__restrict__ o, const bf16 *__restrict__ a,
                            const bf16 *__restrict__ b, long long n8) {
  long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n8;
       i += stride)
    reinterpret_cast<float4 *>(o)[i] =
        add8(reinterpret_cast<const float4 *>(a)[i],
             reinterpret_cast<const float4 *>(b)[i]);
}

// ---------------------------------------------------------------- harness

template <class F>
static float bench_ms(F f, int iters) {
  cudaEvent_t s, e;
  CK(cudaEventCreate(&s));
  CK(cudaEventCreate(&e));
  for (int i = 0; i < 5; ++i) f();
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(s));
  for (int i = 0; i < iters; ++i) f();
  CK(cudaEventRecord(e));
  CK(cudaEventSynchronize(e));
  float ms;
  CK(cudaEventElapsedTime(&ms, s, e));
  CK(cudaEventDestroy(s));
  CK(cudaEventDestroy(e));
  return ms / iters;
}

static void report(const char *name, float ms) {
  double gbps = 3.0 * BYTES / (ms * 1e-3) / 1e9;
  printf("%-16s %8.2f us   %7.1f GB/s\n", name, ms * 1e3, gbps);
}

__global__ void fill(bf16 *p, float v, long long n) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i < n) p[i] = __float2bfloat16(v);
}

int main(int argc, char **argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 100;

  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("%s  sm_%d%d  %d SMs\n", p.name, p.major, p.minor,
         p.multiProcessorCount);

  bf16 *a, *b, *o;
  CK(cudaMalloc(&a, BYTES));
  CK(cudaMalloc(&b, BYTES));
  CK(cudaMalloc(&o, BYTES));
  fill<<<(N + 255) / 256, 256>>>(a, 1.0f, N);
  fill<<<(N + 255) / 256, 256>>>(b, 2.0f, N);
  CK(cudaDeviceSynchronize());

  const int T = 256;
  int gs_blocks = p.multiProcessorCount * 4;  // ~4 waves at 256 thr/blk

  auto k0 = [&] { add_scalar<<<(N + T - 1) / T, T>>>(o, a, b, N); };
  auto k1 = [&] { add_vec8<<<(N8 + T - 1) / T, T>>>(o, a, b, N8); };
  auto k2 = [&] { add_vec8_gs<<<gs_blocks, T>>>(o, a, b, N8); };

  report("add_scalar", bench_ms(k0, iters));
  report("add_vec8", bench_ms(k1, iters));
  report("add_vec8_gs", bench_ms(k2, iters));
  CK(cudaGetLastError());

  // correctness spot check
  bf16 h[4];
  CK(cudaMemcpy(h, o, sizeof(h), cudaMemcpyDeviceToHost));
  printf("check out[0] = %.1f (expect 3.0)\n", __bfloat162float(h[0]));

  // exactly one launch of each under `ncu --profile-from-start off`
  CK(cudaProfilerStart());
  k0();
  k1();
  k2();
  CK(cudaDeviceSynchronize());
  CK(cudaProfilerStop());

  CK(cudaFree(a));
  CK(cudaFree(b));
  CK(cudaFree(o));
  return 0;
}
