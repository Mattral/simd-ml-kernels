#pragma once

#include <cstddef>

namespace simd_ml {
namespace gemm {

// ─── Tile/blocking constants ──────────────────────────────────────────────────
// These constants are derived from typical Intel client CPU cache hierarchy.
// See docs/DESIGN.md §2.3 for where each block is meant to live.
//
//   MR=8, NR=8  — micro-kernel register block: 8 YMM accumulators (one per row),
//                 1 YMM for B panel → 9/16 YMM used, leaves headroom for prefetch
//   KC=256      — K-panel: Ac(MC×KC) + Bc(KC×NC) panel pair fits in L2
//   MC=128      — M-panel: Ac stays L2-resident while B panel streams from L3
//   NC=2048     — N-panel: large to amortise B packing cost across many M-tiles
static constexpr int MR = 8;
static constexpr int NR = 8;
static constexpr int KC = 256;
static constexpr int MC = 128;
static constexpr int NC = 2048;

// ─── Small-matrix dispatch threshold ─────────────────────────────────────────
// When max(M,N,K) ≤ SMALL_GEMM_THRESHOLD, sgemm_packed dispatches to
// sgemm_direct_avx2 which skips panel packing entirely.
//
// Rationale (see docs/DESIGN.md §2.1):
//   At N=64, the KC×NC packing buffer is ≈2 MB regardless of actual N.
//   Allocating and filling it costs more time than the arithmetic itself,
//   so sgemm_packed at N=64 runs at ~0.7× the speed of a plain AVX2 loop.
//   At SMALL_GEMM_THRESHOLD=128, all three matrices (A+B+C ≈ 192 KB) fit
//   comfortably in L2, hardware prefetch handles strided B access, and no
//   packing is needed. Above the threshold, packing amortises quickly and
//   the packed path wins.
static constexpr int SMALL_GEMM_THRESHOLD = 128;

// ─── AVX-512 tile constants ───────────────────────────────────────────────────
// MR512 == MR: pack_a_panel is NR-independent so it is reused verbatim for
// both ISA paths. NR512 = 16 (one full __m512 = 16 floats) — a SINGLE
// accumulator per row, not a dual-half split like avx_matmul.cpp's 6×32
// kernel. This deliberately avoids the "second half never written" bug
// class entirely: there is no second half to forget, by construction.
static constexpr int MR512 = MR;   // = 8
static constexpr int NR512 = 16;

// Reference: Goto, K. & van de Geijn, R. (2008). Anatomy of High-Performance
// Matrix Multiplication. ACM Transactions on Mathematical Software, 34(3).

// ─── Panel packing ────────────────────────────────────────────────────────────

/** pack_b_panel — Reorder B[k × n] into NR-column-major panel layout. */
void pack_b_panel(const float* B, int ldb, int k, int n, float* B_packed);

/** pack_a_panel — Reorder A[m × k] into row-major micro-panel layout. */
void pack_a_panel(const float* A, int lda, int m, int k, float* A_packed);

// ─── AVX2 8×8 micro-kernel ────────────────────────────────────────────────────

/**
 * inner_kernel_8x8 — Compute C[0..7][0..7] += alpha * A_packed × B_packed
 *
 * A_packed: row-major, shape [MR × k_rem]   (from pack_a_panel)
 * B_packed: NR-column-major, shape [k_rem × NR]  (from pack_b_panel)
 * C:        row-major output with leading dimension ldc
 */
void inner_kernel_8x8(const float* A_packed,
                       const float* B_packed,
                       float*       C, int ldc,
                       int          k_rem,
                       float        alpha) noexcept;

// ─── AVX-512 8×16 micro-kernel (parallel path, independent of the AVX2 one) ──

/**
 * pack_b_panel_avx512 — Reorder B[k × n] into NR512-column-major panel layout.
 * Identical in spirit to pack_b_panel, but blocks on NR512=16 instead of NR=8.
 */
void pack_b_panel_avx512(const float* B, int ldb, int k, int n, float* B_packed);

/**
 * inner_kernel_8x16_avx512 — Compute C[0..7][0..15] += alpha * A × B
 *
 * Same broadcast-based structure as inner_kernel_8x8 (8 rows, one accumulator
 * per row), but each accumulator is a full __m512 (16 floats) instead of a
 * __m256 (8 floats) — doubling throughput per k-step with no dual-accumulator
 * complexity, since one ZMM register already spans the full NR512 width.
 */
void inner_kernel_8x16_avx512(const float* A_packed,
                              const float* B_packed,
                              float*       C, int ldc,
                              int          k_rem,
                              float        alpha) noexcept;

// ─── Main GEMM entry point ────────────────────────────────────────────────────

/**
 * sgemm_direct_avx2 — AVX2 GEMM without panel packing (small-matrix fast path)
 *
 * Accesses A and B directly with their original strides rather than packing
 * into contiguous panels. For max(M,N,K) ≤ SMALL_GEMM_THRESHOLD (128), the
 * entire working set fits in L2 cache and hardware prefetch handles the
 * strided B access — making packing overhead a net loss.
 *
 * Exposed publicly (not just internal) so test_gemm_packed.cpp can verify it
 * in isolation, independently of the dispatch path in sgemm_packed.
 */
void sgemm_direct_avx2(int M, int N, int K,
                        float alpha,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float beta,
                        float* C, int ldc) noexcept;

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
 * sgemm_packed_avx2 / sgemm_packed_avx512 — the per-ISA implementations
 * (8×8 and 8×16 micro-kernels). Both use sgemm_direct_avx2 when
 * max(M,N,K) <= SMALL_GEMM_THRESHOLD. Call only on a CPU that has the ISA;
 * prefer kernels_for(Isa::...)->sgemm, which checks.
 */
void sgemm_packed_avx2(int M, int N, int K, float alpha,
                       const float* A, int lda, const float* B, int ldb,
                       float beta, float* C, int ldc);
void sgemm_packed_avx512(int M, int N, int K, float alpha,
                         const float* A, int lda, const float* B, int ldb,
                         float beta, float* C, int ldc);

/** scale_matrix_c — C = beta*C; beta = 0 writes zeros without reading C. AVX2 unit. */
void scale_matrix_c(float* C, int M, int N, int ldc, float beta) noexcept;

// ─── OpenMP thread control ────────────────────────────────────────────────────

/**
 * set_num_threads / get_num_threads — the OpenMP thread count used by the
 * jc-parallel loop of the packed GEMMs (dispatch.cpp). Without
 * -DSIMD_ML_OPENMP, get_num_threads() is always 1.
 */
void set_num_threads(int n) noexcept;
int  get_num_threads()      noexcept;

}  // namespace gemm
}  // namespace simd_ml
