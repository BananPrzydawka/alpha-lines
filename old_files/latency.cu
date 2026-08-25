// ============================================================================
// How long does one GPU instruction take?
//
// There are two different answers, and this file measures both.
//
//   LATENCY    How many cycles until the result is usable by the next
//              instruction. Measured with a chain where every operation
//              depends on the previous one, so nothing can overlap.
//
//   THROUGHPUT How often the pipeline accepts a new operation. Measured with
//              several independent chains running side by side, so the
//              pipeline never has to wait for a result.
//
// Both run a single warp on a single SM, so no other warp can fill in the
// stall cycles and make the numbers look better than they are.
//
// Every operation is written as inline PTX. That matters: an intrinsic like
// __frcp_rn() can expand into a whole sequence of instructions, and then you
// are timing a sequence instead of an instruction. One line of PTX with a
// read-write constraint gives you exactly one machine instruction, and a
// dependency the optimiser is not allowed to remove.
//
// KEEPING THE CHAIN ALIVE
// `asm volatile` binds the frontend only: it guarantees the instruction
// reaches the PTX, but ptxas is then free to constant-fold the whole chain if
// every input traces back to a literal. So the accumulators are seeded from a
// kernel parameter, which is unknowable until launch. That is the only reason
// the chain survives; there is no loop left to hide behind.
//
// Note that this defeats folding, not algebra. add.f32 is not reassociable so
// a float chain has to be executed step by step, but add.s32 is: ptxas may
// legally rewrite N adds of a fixed operand into one multiply. Check the
// integer rows in the disassembly separately.
//
// Always sanity-check with:  make dis F=latency K=lat_fadd
// You should see REPEATS copies of the instruction and nothing else.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>

// How many times the measured instruction is stamped out between the two
// clock reads. Nothing multiplies this any more, so it alone has to amortise
// the cost of reading the clock: too small and that fixed overhead shows up in
// the result. Too large and the unrolled body outgrows the instruction cache,
// at which point you are measuring instruction fetch. Sweep it with
// -DREPEATS=n and check the reported cycles/op is flat.
#ifndef REPEATS
#define REPEATS 4
#endif

// Independent chains in the throughput kernels. 8 is comfortably more than any
// pipeline's latency-to-throughput ratio on Hopper, so the pipe stays full.
#define CHAINS 8

// A result below 1.0 cycles per operation is physically impossible: no unit
// retires more than one warp-instruction per cycle. If you see one, the
// compiler deleted the chain and the number is meaningless.
#define IMPOSSIBLE 1.0

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "%s:%d  %s\n", __FILE__, __LINE__,                       \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Reads %clock64, a counter that ticks once per SM clock cycle. The "memory"
// clobber stops the compiler from moving loads and stores across this point.
#define READ_CLOCK(dest)                                                       \
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(dest)::"memory")

// Bit-copies the 32-bit seed into an accumulator. mov.b32 is a typeless move,
// so the same line works whether the destination is an .f32 or a .u32
// register. This must sit before the first clock read, or the parameter load
// lands inside the timed window.
//
// `cons` has to be a parameter here rather than picked up from the enclosing
// kernel macro: parameter substitution only rewrites tokens that appear in the
// body of the macro being expanded, so a bare `constraint` inside SEED would
// survive as an undeclared identifier.
#define SEED(dest, cons, bits)                                                 \
  asm volatile("mov.b32 %0, %1;" : "=" cons(dest) : "r"(bits))

// ============================================================================
// Kernel templates
//
// These are macros, not functions: the preprocessor stamps out one complete
// __global__ kernel per instruction we want to measure. Run
//   nvcc -E -arch=sm_90a src/latency.cu
// to see the generated code.
//
// Parameters:
//   kernel_name  name of the kernel to generate
//   value_type   C type held in each accumulator (float, unsigned, ...)
//   constraint   inline-asm register class: "f" = 32-bit float register,
//                "r" = 32-bit integer register
//   start_value  what the accumulators are initialised to, passed in at launch
//                as raw bits rather than baked in as a literal
//   instruction  the PTX to measure. %0 is the accumulator (read and written),
//                %1 is a second operand that is only ever read.
// ============================================================================

// One accumulator. Operation N+1 cannot start until operation N has produced
// its result, so the measured time is REPEATS x latency.
#define MAKE_LATENCY_KERNEL(kernel_name, value_type, constraint, start_value,  \
                            instruction)                                       \
  __global__ void kernel_name(long long *cycles_out, void *sink,               \
                              unsigned seed) {                                 \
    value_type acc, operand;                                                   \
    long long start, stop;                                                     \
                                                                               \
    SEED(acc, constraint, seed);                                               \
    SEED(operand, constraint, seed);                                           \
                                                                               \
    READ_CLOCK(start);                                                         \
    _Pragma("unroll") for (int r = 0; r < REPEATS; ++r)                        \
        asm volatile(instruction : "+" constraint(acc) : constraint(operand)); \
    READ_CLOCK(stop);                                                          \
                                                                               \
    /* The store keeps `acc` alive so the chain cannot be optimised away. */   \
    ((value_type *)sink)[threadIdx.x] = acc;                                   \
    if (threadIdx.x == 0) *cycles_out = stop - start;                          \
  }

// CHAINS accumulators that never touch each other. The scheduler can issue one
// per cycle without waiting, so the measured time is REPEATS x CHAINS x
// (cycles the pipeline needs per operation).
#define MAKE_THROUGHPUT_KERNEL(kernel_name, value_type, constraint,            \
                               start_value, instruction)                       \
  __global__ void kernel_name(long long *cycles_out, void *sink,               \
                              unsigned seed) {                                 \
    value_type acc[CHAINS];                                                    \
    value_type operand;                                                        \
    long long start, stop;                                                     \
                                                                               \
    /* seed + c, not seed: CHAINS identical expressions are a CSE target, and  \
       collapsing them would report throughput CHAINS times too fast. One ulp  \
       of difference is enough to keep them distinct. */                       \
    _Pragma("unroll") for (int c = 0; c < CHAINS; ++c)                         \
        SEED(acc[c], constraint, seed + c);                                    \
    SEED(operand, constraint, seed);                                           \
                                                                               \
    READ_CLOCK(start);                                                         \
    _Pragma("unroll") for (int r = 0; r < REPEATS; ++r) {                      \
      _Pragma("unroll") for (int c = 0; c < CHAINS; ++c)                       \
          asm volatile(instruction                                             \
                       : "+" constraint(acc[c])                                \
                       : constraint(operand));                                 \
    }                                                                          \
    READ_CLOCK(stop);                                                          \
                                                                               \
    value_type total = acc[0];                                                 \
    _Pragma("unroll") for (int c = 1; c < CHAINS; ++c) total += acc[c];        \
    ((value_type *)sink)[threadIdx.x] = total;                                 \
    if (threadIdx.x == 0) *cycles_out = stop - start;                          \
  }

// Generate both kernels for one instruction, plus the host-side helper that
// hands start_value to them as raw bits.
#define MEASURE(short_name, value_type, constraint, start_value, instruction)  \
  MAKE_LATENCY_KERNEL(lat_##short_name, value_type, constraint, start_value,   \
                      instruction)                                             \
  MAKE_THROUGHPUT_KERNEL(tpt_##short_name, value_type, constraint,             \
                         start_value, instruction)                             \
  static unsigned seed_##short_name(void) {                                    \
    value_type v = start_value;                                                \
    unsigned bits;                                                             \
    memcpy(&bits, &v, sizeof(bits));                                           \
    return bits;                                                               \
  }

// ============================================================================
// The instructions under test
//
// Note the two forms of reciprocal. rcp.approx.f32 is a single MUFU
// instruction; div.rn.f32 is an IEEE-correct sequence of a dozen or so. The
// gap between them is the whole reason to prefer __frcp_rn's fast cousins in
// a hot loop.
// ============================================================================

// bfloat16 arithmetic needs sm_80 or newer. ptxas parses the whole file
// regardless of which branch runs, so the mnemonic itself has to disappear on
// older targets rather than being guarded at runtime.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define BF16_ADD "add.rn.bf16x2 %0, %0, %1;"
#else
#define BF16_ADD "add.s32 %0, %0, %1;"  // placeholder; result is marked n/a
#endif

// 32-bit float, the FMA pipe
MEASURE(fadd, float, "f", 1.0001f, "add.f32 %0, %0, %1;")
MEASURE(fmul, float, "f", 1.0001f, "mul.f32 %0, %0, %1;")
MEASURE(ffma, float, "f", 1.0001f, "fma.rn.f32 %0, %0, %1, %1;")

// 32-bit integer, the ALU pipe.
// These are the rows to distrust: integer arithmetic is reassociable, so a
// long chain of adds against a fixed operand can legally become one multiply,
// and an even-length xor chain against a fixed operand is the identity. The
// runtime seed does not prevent either. Read the disassembly.
MEASURE(iadd, unsigned, "r", 1u, "add.s32 %0, %0, %1;")
MEASURE(ixor, unsigned, "r", 0x9e3779b9u, "xor.b32 %0, %0, %1;")
MEASURE(imad, unsigned, "r", 1u, "mad.lo.s32 %0, %0, %1, %1;")
MEASURE(ishf, unsigned, "r", 0x9e3779b9u, "shf.l.wrap.b32 %0, %0, %0, %1;")

// packed 16-bit: two values per instruction
MEASURE(f16x2, unsigned, "r", 0x3c003c00u, "add.f16x2 %0, %0, %1;")
MEASURE(bf16x2, unsigned, "r", 0x3f803f80u, BF16_ADD)

// MUFU: the special-function unit. Far narrower than the FMA pipe.
MEASURE(ex2, float, "f", 0.5f, "ex2.approx.f32 %0, %0;")
MEASURE(lg2, float, "f", 1.5f, "lg2.approx.f32 %0, %0;")
MEASURE(rcp, float, "f", 1.0001f, "rcp.approx.f32 %0, %0;")
MEASURE(rsq, float, "f", 1.0001f, "rsqrt.approx.f32 %0, %0;")
MEASURE(sin, float, "f", 0.7f, "sin.approx.f32 %0, %0;")

// Multi-instruction sequences, for contrast. These are NOT single
// instructions; the numbers are the cost of the whole expansion.
MEASURE(fdiv, float, "f", 1.0001f, "div.rn.f32 %0, %0, %1;")
MEASURE(fsqrt, float, "f", 2.0f, "sqrt.rn.f32 %0, %0;")

// ============================================================================
// Host side
// ============================================================================

typedef void (*kernel_ptr)(long long *, void *, unsigned);

static long long *device_cycles;
static void *device_sink;

// Launch once to warm the instruction cache, once for real, return the cycle
// count the kernel wrote. Timing is done on the device with %clock64, so no
// host-side event machinery is involved.
static long long run_kernel(kernel_ptr kernel, unsigned seed) {
  kernel<<<1, 32>>>(device_cycles, device_sink, seed);
  kernel<<<1, 32>>>(device_cycles, device_sink, seed);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());

  long long cycles;
  CHECK(cudaMemcpy(&cycles, device_cycles, sizeof(cycles),
                   cudaMemcpyDeviceToHost));
  return cycles;
}

static void print_row(const char *name, kernel_ptr latency_kernel,
                      kernel_ptr throughput_kernel, unsigned seed,
                      bool supported) {
  if (!supported) {
    printf("  %-8s %14s %14s   %s\n", name, "n/a", "n/a", "needs sm_80+");
    return;
  }

  double latency_ops = REPEATS;
  double throughput_ops = (double)REPEATS * CHAINS;

  double latency = run_kernel(latency_kernel, seed) / latency_ops;
  double throughput = run_kernel(throughput_kernel, seed) / throughput_ops;

  const char *warning = "";
  if (latency < IMPOSSIBLE || throughput < IMPOSSIBLE)
    warning = "  <-- below 1 cyc/op: chain was optimised away, ignore";

  printf("  %-8s %10.2f cyc %10.2f cyc%s\n", name, latency, throughput,
         warning);
}

int main(void) {
  cudaDeviceProp props;
  CHECK(cudaGetDeviceProperties(&props, 0));
  bool has_bf16 = props.major >= 8;

  printf("%s  sm_%d%d\n", props.name, props.major, props.minor);
  printf("%d repeats, %d independent chains for throughput\n\n", REPEATS,
         CHAINS);
  printf("  %-8s %14s %14s\n", "op", "latency", "throughput");
  printf("  %-8s %14s %14s\n", "", "(1 chain)", "(per op)");

  CHECK(cudaMalloc(&device_cycles, sizeof(long long)));
  CHECK(cudaMalloc(&device_sink, 64 * sizeof(double)));

#define ROW(short_name)                                                        \
  print_row(#short_name, lat_##short_name, tpt_##short_name,                   \
            seed_##short_name(), true)
#define ROW_SM80(short_name)                                                   \
  print_row(#short_name, lat_##short_name, tpt_##short_name,                   \
            seed_##short_name(), has_bf16)

  puts("\n  -- fp32 --");
  ROW(fadd); ROW(fmul); ROW(ffma);

  puts("\n  -- int32 --");
  ROW(iadd); ROW(ixor); ROW(imad); ROW(ishf);

  puts("\n  -- packed 16-bit --");
  ROW(f16x2); ROW_SM80(bf16x2);

  puts("\n  -- MUFU (special function unit) --");
  ROW(ex2); ROW(lg2); ROW(rcp); ROW(rsq); ROW(sin);

  puts("\n  -- multi-instruction sequences, not single ops --");
  ROW(fdiv); ROW(fsqrt);

#undef ROW
#undef ROW_SM80

  CHECK(cudaFree(device_cycles));
  CHECK(cudaFree(device_sink));
  return 0;
}
