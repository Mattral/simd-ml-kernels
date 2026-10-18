#pragma once

#include <cstddef>

namespace simd_ml {
namespace gemm {

// ─── AVX2 FP32 GEMM (avx2_gemm_packed.cpp) ────────────────────────────────────
// One 6×16 micro-kernel (12 YMM accumulators), used two ways:
//   - direct: on A and B in place, for products small enough that packing
//     would cost more than it saves (sgemm_avx2_prefers_direct);
//   - packed: Goto/BLIS five loops, A and B packed into micro-panels, block
//     sizes KC/MC/NC derived at run time from the caches (CPUID leaf 4,
//     dispatch::gemm_blocking).
// The shape, the block-size rules and the direct/packed cut-off are justified
// with measurements in avx2_gemm_packed.cpp and docs/DESIGN.md §2.
static constexpr int MR_AVX2 = 6;
static constexpr int NR_AVX2 = 16;

/**
 * sgemm_direct_avx2 — the unpacked path: C = alpha*A*B + beta*C with the
 * 6×16 kernel reading A and B with their own strides. Public so the tests can
 * run it on sizes the dispatcher would send to the packed path.
 */
void sgemm_direct_avx2(int M, int N, int K,
                       float alpha,
                       const float* A, int lda,
                       const float* B, int ldb,
                       float beta,
                       float* C, int ldc) noexcept;

/** Whether sgemm_packed_avx2 takes the direct path for this shape. */
bool sgemm_avx2_prefers_direct(int M, int N, int K) noexcept;

/**
 * sgemm_packed — C = alpha*A*B + beta*C (float32, row-major), through the
 * kernel table chosen for this CPU (dispatch/kernels.hpp); defined in
 * dispatch.cpp, so it can be called from code built for any CPU.
 */
void sgemm_packed(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc);

/**
 * sgemm_packed_avx2 / sgemm_packed_avx512 — the per-ISA implementations.
 * Call only on a CPU that has the ISA; prefer kernels_for(Isa::...)->sgemm,
 * which checks.
 */
void sgemm_packed_avx2(int M, int N, int K, float alpha,
                       const float* A, int lda, const float* B, int ldb,
                       float beta, float* C, int ldc);
void sgemm_packed_avx512(int M, int N, int K, float alpha,
                         const float* A, int lda, const float* B, int ldb,
                         float beta, float* C, int ldc);

/** KC/MC/NC the AVX2 path uses on this machine (derived once, on first use). */
void sgemm_blocking_avx2(int* kc, int* mc, int* nc) noexcept;

/** scale_matrix_c — C = beta*C; beta = 0 writes zeros without reading C. AVX2 unit. */
void scale_matrix_c(float* C, int M, int N, int ldc, float beta) noexcept;

// ─── AVX-512 path (avx512_gemm_packed.cpp) ────────────────────────────────────
// 8×16 kernel with fixed blocking; its own packing.
static constexpr int MR512 = 8;
static constexpr int NR512 = 16;
static constexpr int KC512 = 256;
static constexpr int MC512 = 128;
static constexpr int NC512 = 2048;

/** KC/MC/NC of the AVX-512 path. */
void sgemm_blocking_avx512(int* kc, int* mc, int* nc) noexcept;

// ─── OpenMP thread control ────────────────────────────────────────────────────

/**
 * set_num_threads / get_num_threads — the OpenMP thread count used by the
 * jc-parallel loop of the packed GEMMs (dispatch.cpp). Without
 * -DSIMD_ML_OPENMP, get_num_threads() is always 1.
 */
void set_num_threads(int n) noexcept;
int  get_num_threads()      noexcept;

// Reference: Goto, K. & van de Geijn, R. (2008). Anatomy of High-Performance
// Matrix Multiplication. ACM TOMS 34(3). Low, T. M. et al. (2016). Analytical
// Modeling Is Enough for High-Performance BLIS. ACM TOMS 43(2).

}  // namespace gemm
}  // namespace simd_ml
