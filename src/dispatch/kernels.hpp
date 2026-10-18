#pragma once

/**
 * kernels.hpp — one table of kernel entry points per instruction set, and the
 * choice of table for this CPU.
 *
 * Each ISA lives in its own translation units, compiled with that ISA's flags
 * and nothing more (CMakeLists.txt, "ISA units"):
 *
 *   scalar   baseline x86-64 (SSE2) or the target's default   always built
 *   avx2     -mavx2 -mfma -mf16c                                x86 only
 *   avx512   -mavx512f -mavx512dq -mavx512bw -mavx512vl          x86 only
 *
 * Everything else (this dispatch code, the Python bindings, the tests' own
 * code) is compiled for the baseline, so the library loads and runs on any
 * x86-64 CPU; AVX2/AVX-512 instructions execute only after the CPU and the OS
 * were checked (cpuid.hpp). A table may reuse another table's functions: the
 * AVX-512 table points at the AVX2 elementwise kernels, because only GEMM has
 * an AVX-512 path.
 *
 * Selection, made once on first use: the widest table whose requirements the
 * CPU meets (Isa order below), unless the environment variable SIMD_ML_ISA is
 * "scalar", "avx2" or "avx512", which caps the choice at that table (useful to
 * test or benchmark a narrower path on a wider CPU; asking for an ISA the CPU
 * lacks falls back to the widest supported one below it).
 */

#include <cstddef>
#include <cstdint>

#include "cpuid.hpp"

namespace simd_ml {

enum class Isa : int { scalar = 0, avx2 = 1, avx512 = 2 };
constexpr int kNumIsa = 3;

/** C = alpha·op(A)·op(B) + beta·C; trans_a: A stored K×M, trans_b: B stored N×K. */
using SgemmFn  = void (*)(bool trans_a, bool trans_b, int M, int N, int K, float alpha,
                          const float* A, int lda, const float* B, int ldb,
                          float beta, float* C, int ldc);
using Sgemm16Fn = void (*)(int M, int N, int K, float alpha,
                           const std::uint16_t* A, int lda, const std::uint16_t* B, int ldb,
                           float beta, float* C, int ldc);
/** C[i][j] = a_scale·b_scales[j]·Σ_k (A[i][k] − a_zero_point)·op(B)[k][j]; gemm/int8_gemm.hpp. */
using GemmU8S8Fn = void (*)(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                            int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                            const float* b_scales, float* C, int ldc);
using UnaryFn  = void (*)(const float* in, float* out, std::size_t n);
using RowFn    = void (*)(const float* in, float* out, int n);
using LayerNormFn = void (*)(const float* in, float* out, int n,
                             const float* gamma, const float* beta, float eps);
using BlockingFn = void (*)(int* kc, int* mc, int* nc);

struct KernelTable {
    Isa         isa;
    const char* name;        ///< "scalar", "avx2", "avx512"
    const char* flags;       ///< compiler flags of this ISA's units
    const char* sgemm_tile;  ///< micro-kernel shape, for build_info()
    BlockingFn  sgemm_blocking;  ///< KC/MC/NC in use (0 = not blocked), for build_info()
    SgemmFn     sgemm;       ///< C = alpha*op(A)*op(B) + beta*C, row-major FP32
    Sgemm16Fn   sgemm_f16;   ///< IEEE binary16 inputs, FP32 accumulate and output
    Sgemm16Fn   sgemm_bf16;  ///< BF16 bit patterns, FP32 accumulate and output
    GemmU8S8Fn  sgemm_u8s8;  ///< uint8 × int8 with per-column scales, FP32 output
    UnaryFn     gelu_erf;    ///< exact GeLU; in == out allowed
    UnaryFn     gelu_tanh;   ///< tanh-approximate GeLU; in == out allowed
    RowFn       relu;
    RowFn       silu;
    RowFn       softmax_row;
    LayerNormFn layer_norm_row;
};

/** CPUID/XGETBV results, detected once. */
const dispatch::CpuFeatures& cpu_features() noexcept;

/** The table used by the Python bindings and the dispatching C++ entry points. */
const KernelTable& kernels() noexcept;

/**
 * use_kernels — make `isa`'s table the active one for the whole process, if
 * kernels_for(isa) is non-null; returns whether it did. For tests and
 * selftest: not synchronised with calls in flight on other threads.
 */
bool use_kernels(Isa isa) noexcept;

/**
 * kernels_for — the table for one ISA, or nullptr if it was not compiled into
 * this build or this CPU/OS cannot run it. Ignores SIMD_ML_ISA. Tests use it
 * to run every path the machine supports.
 */
const KernelTable* kernels_for(Isa isa) noexcept;

/** Whether `isa`'s units were compiled into this build (regardless of the CPU). */
bool isa_compiled(Isa isa) noexcept;

const char* isa_name(Isa isa) noexcept;

// Per-ISA tables, defined in each ISA's units. Call kernels_for(), not these:
// calling them is safe, but running a function from the table is not unless
// the CPU supports the ISA.
const KernelTable& kernel_table_scalar() noexcept;
const KernelTable& kernel_table_avx2() noexcept;
const KernelTable& kernel_table_avx512() noexcept;

}  // namespace simd_ml
