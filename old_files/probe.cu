// Timing probes: kernel launch overhead and memory-hierarchy latency.
//
//   ./probe            # default sweep
//   ./probe 20000      # more launch iterations for the cheapest configs

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,                        \
              cudaGetErrorString(e_));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------- launch cost

__global__ void empty() {}
__global__ void empty_args(float *a, float *b, float *c, long long n) {}

#define MAX_BLOCKS_LOG2 16
#define MAX_THREADS 1024

// Total thread-slots to spend on any single cell. Small grids get the full
// iteration count; big ones would otherwise dominate the whole run, so the
// count is scaled down until the product fits this budget.
#define WORK_BUDGET (1LL << 24)
#define MIN_ITERS 30

static int iters_for(int blocks, int threads, int base) {
  long long n = WORK_BUDGET / ((long long)blocks * threads);
  if (n > base) n = base;
  if (n < MIN_ITERS) n = MIN_ITERS;
  return (int)n;
}

// Microseconds per launch, measured back to back with no sync in the loop, so
// this is the rate at which the driver and the GPU's work distributor can
// absorb launches rather than the latency of any one of them.
static double bench(int blocks, int threads, int iters, bool with_args) {
  cudaEvent_t s, e;
  CK(cudaEventCreate(&s));
  CK(cudaEventCreate(&e));

  for (int i = 0; i < 3; ++i) {
    if (with_args) empty_args<<<blocks, threads>>>(nullptr, nullptr, nullptr, 0);
    else empty<<<blocks, threads>>>();
  }
  CK(cudaDeviceSynchronize());

  CK(cudaEventRecord(s));
  for (int i = 0; i < iters; ++i) {
    if (with_args) empty_args<<<blocks, threads>>>(nullptr, nullptr, nullptr, 0);
    else empty<<<blocks, threads>>>();
  }
  CK(cudaEventRecord(e));
  CK(cudaEventSynchronize(e));
  CK(cudaGetLastError());

  float ms;
  CK(cudaEventElapsedTime(&ms, s, e));
  CK(cudaEventDestroy(s));
  CK(cudaEventDestroy(e));
  return ms * 1e3 / iters;
}

static void sweep(const char *label, bool with_args, int base_iters,
                  int max_threads) {
  printf("\n%s -- us per launch\n\n", label);

  printf("%10s", "blocks\\thr");
  for (int t = 1; t <= max_threads; t *= 2) printf("%9d", t);
  putchar('\n');

  for (int b = 1; b <= (1 << MAX_BLOCKS_LOG2); b *= 2) {
    printf("%10d", b);
    for (int t = 1; t <= max_threads; t *= 2)
      printf("%9.3f", bench(b, t, iters_for(b, t, base_iters), with_args));
    putchar('\n');
    fflush(stdout);
  }
}

// ------------------------------------------------------- memory latency chase
//
// One dependent load at a time: the address of load N+1 comes from the data of
// load N, so nothing can be prefetched or overlapped and the memory pipeline is
// never more than one request deep. cycles/step is then the load-to-use latency
// of a single load at that working-set size.
//
// LAYOUT
// The buffer is a linked cycle with one node per 128-byte line, and each node
// holds the word index of the next. The link lives in the node's first word, so
// the same buffer serves every load width: a 32-bit load reads just the link, a
// 128-bit load also pulls the three padding words beside it. That keeps one
// step per cache line in all three cases, which is what makes the widths
// comparable.
//
// The order is a random permutation of the nodes rather than ascending. A
// linear walk is friendly to DRAM row locality and to the TLB, and would
// under-report latency at the large sizes where those effects dominate.
//
// WIDTH
// 128 bits is the widest single-instruction load the ISA has (LDG.E.128), so
// the sweep stops there. The loads are written as inline PTX because reading
// only the first component of a wider load lets the compiler narrow it back
// down to 32 bits, which would silently measure the same thing three times.

#define WORDS_PER_LINE 32       // 128 B / 4 B
#define MIN_STEPS 4096
#define MAX_STEPS (1 << 20)     // caps runtime; the walk is random, so a
                                // partial lap still samples the whole buffer

// The unused components go into `junk`, which is stored at the end so the wide
// load cannot be narrowed. That XOR chain is independent of the address chain,
// so it issues during the load stall and does not lengthen the critical path.
#define CHASE(load_stmt)                                                       \
  uint32_t p = 0, junk = 0;                                                    \
  for (int i = 0; i < warm; ++i) { load_stmt; }                                \
  __syncwarp();                                                                \
  long long t0 = clock64();                                                    \
  for (int i = 0; i < steps; ++i) { load_stmt; }                               \
  long long t1 = clock64();                                                    \
  sink[0] = p;                                                                 \
  sink[1] = junk;                                                              \
  *cyc = t1 - t0;

__global__ void chase32(const uint32_t *buf, int steps, int warm,
                        uint32_t *sink, long long *cyc) {
  CHASE(asm volatile("ld.global.u32 %0, [%1];"
                     : "=r"(p) : "l"(buf + p) : "memory"))
}

__global__ void chase64(const uint32_t *buf, int steps, int warm,
                        uint32_t *sink, long long *cyc) {
  uint32_t b;
  CHASE(asm volatile("ld.global.v2.u32 {%0, %1}, [%2];"
                     : "=r"(p), "=r"(b) : "l"(buf + p) : "memory");
        junk ^= b)
}

__global__ void chase128(const uint32_t *buf, int steps, int warm,
                         uint32_t *sink, long long *cyc) {
  uint32_t b, c, d;
  CHASE(asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(p), "=r"(b), "=r"(c), "=r"(d)
                     : "l"(buf + p) : "memory");
        junk ^= b ^ c ^ d)
}

typedef void (*chase_ptr)(const uint32_t *, int, int, uint32_t *, long long *);

static void mem_latency() {
  size_t sizes_kb[] = {2, 8, 16, 32, 64, 128, 256, 512,
                       1024, 4096, 16384, 32768, 65536, 131072, 262144};
  chase_ptr kernels[] = {chase32, chase64, chase128};

  uint32_t *sink;
  long long *cyc, h;
  CK(cudaMalloc(&sink, 8));
  CK(cudaMalloc(&cyc, 8));

  printf("\nload-to-use latency -- cycles per dependent load\n\n");
  printf("%12s %10s %10s %10s\n", "working set", "32-bit", "64-bit", "128-bit");

  std::mt19937 rng(12345);
  for (size_t kb : sizes_kb) {
    size_t n = kb * 1024 / sizeof(uint32_t);
    size_t nodes = n / WORDS_PER_LINE;
    if (nodes < 2) continue;

    // Random cycle through every line, starting at node 0 so the chase can
    // begin at p = 0. Shuffling from index 1 keeps node 0 in place.
    std::vector<uint32_t> order(nodes);
    for (size_t i = 0; i < nodes; ++i) order[i] = (uint32_t)i;
    std::shuffle(order.begin() + 1, order.end(), rng);

    std::vector<uint32_t> host(n, 0);
    for (size_t i = 0; i < nodes; ++i)
      host[(size_t)order[i] * WORDS_PER_LINE] =
          order[(i + 1) % nodes] * WORDS_PER_LINE;

    uint32_t *buf;
    if (cudaMalloc(&buf, n * 4) != cudaSuccess) break;
    CK(cudaMemcpy(buf, host.data(), n * 4, cudaMemcpyHostToDevice));

    int steps = (int)std::min<size_t>(std::max<size_t>(nodes, MIN_STEPS),
                                      MAX_STEPS);
    int warm = (int)std::min<size_t>(nodes, 1 << 16);

    if (kb < 1024) printf("%9zu KB", kb);
    else           printf("%9zu MB", kb / 1024);

    for (chase_ptr k : kernels) {
      k<<<1, 1>>>(buf, steps, warm, sink, cyc);
      CK(cudaDeviceSynchronize());
      CK(cudaGetLastError());
      CK(cudaMemcpy(&h, cyc, 8, cudaMemcpyDeviceToHost));
      printf("%10.1f", (double)h / steps);
    }
    putchar('\n');
    fflush(stdout);

    CK(cudaFree(buf));
  }
  CK(cudaFree(sink));
  CK(cudaFree(cyc));
}

int main(int argc, char **argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 10000;
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("%s  sm_%d%d  %d SMs  L2 %.0f MB\n", p.name, p.major, p.minor,
         p.multiProcessorCount, p.l2CacheSize / 1048576.0);

  int max_threads = p.maxThreadsPerBlock;
  if (max_threads > MAX_THREADS) max_threads = MAX_THREADS;

  sweep("no args", false, iters, max_threads);
  sweep("4 args (3 pointers + long long)", true, iters, max_threads);

  mem_latency();
  return 0;
}
