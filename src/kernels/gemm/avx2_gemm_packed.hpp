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

// Every FP32 GEMM entry point computes C = alpha·op(A)·op(B) + beta·C,
// row-major, op(X) = X or Xᵀ (BLAS semantics): with trans_a A is stored K×M
// (lda >= M), with trans_b B is stored N×K (ldb >= K). Transposes are handled
// while packing (or by the unpacked paths' strides), never by a copy.

/**
 * sgemm_direct_avx2 — the unpacked paths: the 6×16 kernel on op(A) and B in
 * place, or for trans_b dot products along rows of A and B. Public so the
 * tests can run it on shapes the dispatcher would send to the packed path.
 */
void sgemm_direct_avx2(bool trans_a, bool trans_b, int M, int N, int K,
                       float alpha,
                       const float* A, int lda,
                       const float* B, int ldb,
                       float beta,
                       float* C, int ldc);

/**
 * kDotRowsMax — with trans_b (B stored N×K, x·Wᵀ) and at most this many rows
 * of op(A), every table's FP32/FP16/BF16 GEMM takes the row-streaming dot
 * path (the decode GEMV, DESIGN.md §2.7) instead of packing B; the cut-off is
 * measured in avx2_gemm_packed.cpp (choose_path).
 */
static constexpr int kDotRowsMax = 8;

/** Whether sgemm_packed_avx2 takes the direct path for this shape. */
bool sgemm_avx2_prefers_direct(int M, int N, int K) noexcept;

/**
 * sgemm_packed — C = alpha*A*B + beta*C (float32, row-major, no transposes),
 * through the kernel table chosen for this CPU (dispatch/kernels.hpp);
 * defined in dispatch.cpp, so it can be called from code built for any CPU.
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
void sgemm_packed_avx2(bool trans_a, bool trans_b, int M, int N, int K, float alpha,
                       const float* A, int lda, const float* B, int ldb,
                       float beta, float* C, int ldc);
void sgemm_packed_avx512(bool trans_a, bool trans_b, int M, int N, int K, float alpha,
                         const float* A, int lda, const float* B, int ldb,
                         float beta, float* C, int ldc);

/** KC/MC/NC the AVX2 path uses on this machine (derived once, on first use). */
void sgemm_blocking_avx2(int* kc, int* mc, int* nc) noexcept;

/** scale_matrix_c — C = beta*C; beta = 0 writes zeros without reading C. AVX2 unit. */
void scale_matrix_c(float* C, int M, int N, int ldc, float beta) noexcept;

// ─── AVX-512 path (avx512_gemm_packed.cpp) ────────────────────────────────────
// 12×32 micro-kernel (24 ZMM accumulators), same five loops and blocking rules.
static constexpr int MR512 = 12;
static constexpr int NR512 = 32;

/** KC/MC/NC of the AVX-512 path. */
void sgemm_blocking_avx512(int* kc, int* mc, int* nc) noexcept;

// ─── OpenMP thread control ────────────────────────────────────────────────────

/**
 * set_num_threads / get_num_threads — the most OpenMP threads one packed
 * GEMM call uses (dispatch.cpp). Until set_num_threads is called this is
 * OpenMP's default (OMP_NUM_THREADS, else one per logical CPU). Without
 * -DSIMD_ML_OPENMP, get_num_threads() is always 1.
 */
void set_num_threads(int n) noexcept;
int  get_num_threads()      noexcept;

/**
 * Threaded packed GEMM (the loops are in each ISA unit). All threads of one
 * call work on the same NC-wide column block: they pack its KC×NC B panel
 * together (one shared copy in the last-level cache), then take chunks of the
 * block (dynamic schedule), packing each chunk's rows of A into their own
 * buffer. Every C element is computed by one thread in the same order as
 * with one thread, so the result is bitwise the same for any thread count.
 *
 * GemmChunks — an M×N column block cut into m_chunks × n_chunks chunks of
 * mc rows (a multiple of MR, <= MC) by nc columns (a multiple of NR); chunk
 * c covers rows (c / n_chunks)·mc and columns (c % n_chunks)·nc onwards.
 */
struct GemmChunks { int mc, nc, m_chunks, n_chunks; };

/**
 * gemm_chunks — about kGemmChunksPerThread chunks per thread, cut along M
 * first (no A row is packed twice); along N too when M has too few MR-row
 * micro-panels, never narrower than kGemmMinChunkCols (dispatch.cpp).
 */
GemmChunks gemm_chunks(int threads, int M, int N, int mr, int nr, int mc_max) noexcept;

/**
 * gemm_threads — how many threads a packed GEMM of this shape uses:
 * get_num_threads(), fewer for small products (each thread gets at least
 * kGemmMinFlopsPerThread), and 1 inside an enclosing OpenMP parallel region.
 * Always 1 without -DSIMD_ML_OPENMP.
 */
int gemm_threads(int M, int N, int K, int mr, int nr) noexcept;

/**
 * Threaded decode GEMV (x·Wᵀ with few rows of x: the FP32/FP16/BF16 dot path
 * and gemv_q8_0 / gemv_q4_0). The N output rows are cut into one contiguous
 * slice per thread, each a whole number of kGemvRowGroup-row groups (the
 * kernels' row blocks), so every output is computed by one thread with the
 * same code and order as with one thread: bitwise the same for any count.
 *
 * gemv_threads — how many threads a GEMV over N rows reading `bytes` of
 * weights uses: get_num_threads(), fewer so that each thread reads at least
 * kGemvMinBytesPerThread (dispatch.cpp) and has at least one row group, and
 * 1 inside an enclosing OpenMP parallel region. Always 1 without
 * -DSIMD_ML_OPENMP.
 *
 * gemv_slice — rows [lo, hi) of thread t of a team of `team`: the row groups
 * split as evenly as possible (thread sizes differ by at most one group).
 */
constexpr int kGemvRowGroup = 8;
struct GemvSlice { int lo, hi; };
int gemv_threads(int N, double bytes) noexcept;
GemvSlice gemv_slice(int N, int t, int team) noexcept;
/** Tests only: set the minimum weight bytes per thread; returns the previous value. */
double set_gemv_min_bytes_per_thread(double bytes) noexcept;

// Reference: Goto, K. & van de Geijn, R. (2008). Anatomy of High-Performance
// Matrix Multiplication. ACM TOMS 34(3). Low, T. M. et al. (2016). Analytical
// Modeling Is Enough for High-Performance BLIS. ACM TOMS 43(2).

}  // namespace gemm
}  // namespace simd_ml
