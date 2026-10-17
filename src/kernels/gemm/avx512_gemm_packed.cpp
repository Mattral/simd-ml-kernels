/**
 * avx512_gemm_packed.cpp — AVX-512 path of sgemm: the 8×16 micro-kernel and
 * its NR = 16 packing, inside the same Goto/BLIS 5 loops as the AVX2 path
 * (avx2_gemm_packed.cpp; docs/DESIGN.md §2).
 *
 * An AVX-512 unit: compiled with -mavx512f -mavx512dq -mavx512bw -mavx512vl
 * and used only when dispatch.cpp has found those and OS ZMM state. Small
 * matrices (max(M, N, K) <= SMALL_GEMM_THRESHOLD) use the AVX2 direct path,
 * which is in an AVX2 unit; any CPU that runs this one also runs that.
 */

#include "avx2_gemm_packed.hpp"
#include "cache_alloc.hpp"

#include <algorithm>
#include <cstddef>

#if !defined(__AVX512F__)
#  error "avx512_gemm_packed.cpp is an AVX-512 unit: compile with -mavx512f (CMakeLists.txt)"
#endif
#include <immintrin.h>

namespace simd_ml {
namespace gemm {

/**
 * pack_b_panel_avx512 — AVX-512 counterpart of pack_b_panel, blocking on
 * NR512=16 instead of NR=8. Fully independent implementation (not
 * parameterized sharing) to keep the proven AVX2 path completely untouched.
 */
void pack_b_panel_avx512(const float* B, int ldb, int k, int n, float* B_packed) {
    for (int jc = 0; jc < n; jc += NR512) {
        int nr = std::min(NR512, n - jc);
        float* out_block = B_packed + (jc / NR512) * k * NR512;
        for (int p = 0; p < k; ++p) {
            const float* src = B + p * ldb + jc;
            float* dst = out_block + p * NR512;
            if (nr == NR512) {
                _mm512_storeu_ps(dst, _mm512_loadu_ps(src));
            } else {
                for (int j = 0; j < nr; ++j) dst[j] = src[j];
                for (int j = nr; j < NR512; ++j) dst[j] = 0.0f;
            }
        }
    }
}

/**
 * gemm_scalar_block_avx512 — AVX-512-path counterpart of gemm_scalar_block.
 * Identical logic, but reads the B_packed stride as NR512 (16) rather than
 * NR (8), since B_packed for this path was produced by pack_b_panel_avx512.
 * Deliberately duplicated rather than parameterized — this function is only
 * ever hit for small edge/tail blocks, so the tiny code duplication costs
 * nothing at runtime and keeps both ISA paths fully independent to verify.
 */
static void gemm_scalar_block_avx512(int mr, int nr, int kc,
                                     const float* __restrict__ A_packed,
                                     const float* __restrict__ B_packed,
                                     float*       __restrict__ C, int ldc,
                                     float alpha) noexcept {
    for (int i = 0; i < mr; ++i) {
        float* c_row = C + i * ldc;
        const float* a_row = A_packed + i * kc;
        for (int p = 0; p < kc; ++p) {
            float a_val = a_row[p];
            const float* b_col = B_packed + p * NR512;
            for (int j = 0; j < nr; ++j) {
                c_row[j] += alpha * (a_val * b_col[j]);
            }
        }
    }
}

// ─── 8×16 AVX-512 Micro-kernel ───────────────────────────────────────────────
/**
 * inner_kernel_8x16_avx512 — Compute C[0..7][0..15] += alpha * A × B
 *
 * Exact structural mirror of inner_kernel_8x8: 8 rows, ONE accumulator per
 * row, broadcast-based FMA. The only difference is register width: __m512
 * (16 floats) instead of __m256 (8 floats). Because a single ZMM register
 * already spans the full NR512=16 tile width, there is no dual-accumulator
 * split needed — unlike avx_matmul.cpp's 6×32 kernel (which needed 32-wide
 * tiles = 2 ZMM per row). This kernel has no "second half" to forget.
 *
 * Register budget (32 ZMM available): 8 accumulators + 1 B load + transient
 * broadcast = 9 in flight at any moment, 23 registers of headroom.
 */
void inner_kernel_8x16_avx512(const float* __restrict__ A_packed,
                              const float* __restrict__ B_packed,
                              float*       __restrict__ C, int ldc,
                              int k_rem, float alpha) noexcept {
    __m512 c0 = _mm512_loadu_ps(C + 0 * ldc);
    __m512 c1 = _mm512_loadu_ps(C + 1 * ldc);
    __m512 c2 = _mm512_loadu_ps(C + 2 * ldc);
    __m512 c3 = _mm512_loadu_ps(C + 3 * ldc);
    __m512 c4 = _mm512_loadu_ps(C + 4 * ldc);
    __m512 c5 = _mm512_loadu_ps(C + 5 * ldc);
    __m512 c6 = _mm512_loadu_ps(C + 6 * ldc);
    __m512 c7 = _mm512_loadu_ps(C + 7 * ldc);

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();

    const float* a0 = A_packed + 0 * k_rem;
    const float* a1 = A_packed + 1 * k_rem;
    const float* a2 = A_packed + 2 * k_rem;
    const float* a3 = A_packed + 3 * k_rem;
    const float* a4 = A_packed + 4 * k_rem;
    const float* a5 = A_packed + 5 * k_rem;
    const float* a6 = A_packed + 6 * k_rem;
    const float* a7 = A_packed + 7 * k_rem;

    const float* b_ptr = B_packed;

    for (int p = 0; p < k_rem; ++p) {
        __m512 b = _mm512_loadu_ps(b_ptr);  b_ptr += NR512;
        acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a0[p]), b, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a1[p]), b, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a2[p]), b, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a3[p]), b, acc3);
        acc4 = _mm512_fmadd_ps(_mm512_set1_ps(a4[p]), b, acc4);
        acc5 = _mm512_fmadd_ps(_mm512_set1_ps(a5[p]), b, acc5);
        acc6 = _mm512_fmadd_ps(_mm512_set1_ps(a6[p]), b, acc6);
        acc7 = _mm512_fmadd_ps(_mm512_set1_ps(a7[p]), b, acc7);
    }

    if (alpha != 1.0f) {
        __m512 av = _mm512_set1_ps(alpha);
        acc0 = _mm512_mul_ps(acc0, av);  acc1 = _mm512_mul_ps(acc1, av);
        acc2 = _mm512_mul_ps(acc2, av);  acc3 = _mm512_mul_ps(acc3, av);
        acc4 = _mm512_mul_ps(acc4, av);  acc5 = _mm512_mul_ps(acc5, av);
        acc6 = _mm512_mul_ps(acc6, av);  acc7 = _mm512_mul_ps(acc7, av);
    }

    _mm512_storeu_ps(C + 0 * ldc, _mm512_add_ps(c0, acc0));
    _mm512_storeu_ps(C + 1 * ldc, _mm512_add_ps(c1, acc1));
    _mm512_storeu_ps(C + 2 * ldc, _mm512_add_ps(c2, acc2));
    _mm512_storeu_ps(C + 3 * ldc, _mm512_add_ps(c3, acc3));
    _mm512_storeu_ps(C + 4 * ldc, _mm512_add_ps(c4, acc4));
    _mm512_storeu_ps(C + 5 * ldc, _mm512_add_ps(c5, acc5));
    _mm512_storeu_ps(C + 6 * ldc, _mm512_add_ps(c6, acc6));
    _mm512_storeu_ps(C + 7 * ldc, _mm512_add_ps(c7, acc7));
}

// ─── sgemm_packed_avx512 ─────────────────────────────────────────────────────
void sgemm_packed_avx512(int M, int N, int K,
                         float alpha,
                         const float* __restrict__ A, int lda,
                         const float* __restrict__ B, int ldb,
                         float beta,
                         float*       __restrict__ C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    if (std::max({M, N, K}) <= SMALL_GEMM_THRESHOLD) {
        sgemm_direct_avx2(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    scale_matrix_c(C, M, N, ldc, beta);

    const std::size_t b_buf_size_512 = static_cast<std::size_t>(KC) *
        static_cast<std::size_t>((NC + NR512 - 1) / NR512) * NR512;
    const std::size_t a_buf_size = static_cast<std::size_t>(MC) *
        static_cast<std::size_t>(KC);   // MR512 == MR, so A buffer is identical

#ifdef SIMD_ML_OPENMP
    struct ThreadBuffersAVX512 {
        ThreadBuffersAVX512()
            : B_packed(make_aligned_array<float>(
                  static_cast<std::size_t>(KC) *
                  static_cast<std::size_t>((NC + NR512 - 1) / NR512) * NR512)),
              A_packed(make_aligned_array<float>(
                  static_cast<std::size_t>(MC) *
                  static_cast<std::size_t>(KC))) {}
        AlignedUniquePtr<float> B_packed;
        AlignedUniquePtr<float> A_packed;
    };
    thread_local ThreadBuffersAVX512 tls_bufs512;

#pragma omp parallel for schedule(dynamic, 1) \
num_threads(get_num_threads()) if (get_num_threads() > 1)
    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(N - jc, NC);
        float* B_packed = tls_bufs512.B_packed.get();
        float* A_packed = tls_bufs512.A_packed.get();

        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(K - pc, KC);
            pack_b_panel_avx512(B + static_cast<std::size_t>(pc) * ldb + jc,
                                ldb, kc, nc, B_packed);

            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(M - ic, MC);
                pack_a_panel(A + static_cast<std::size_t>(ic) * lda + pc,
                             lda, mc, kc, A_packed);

                for (int jr = 0; jr < nc; jr += NR512) {
                    int nr = std::min(nc - jr, NR512);
                    const float* B_block = B_packed +
                        static_cast<std::size_t>(jr / NR512) * kc * NR512;
                    for (int ir = 0; ir < mc; ir += MR512) {
                        int mr = std::min(mc - ir, MR512);
                        const float* A_block = A_packed +
                            static_cast<std::size_t>(ir) * kc;
                        float* C_block = C + static_cast<std::size_t>(ic + ir) * ldc
                                           + jc + jr;
                        if (mr == MR512 && nr == NR512) {
                            inner_kernel_8x16_avx512(A_block, B_block, C_block, ldc, kc, alpha);
                        } else {
                            gemm_scalar_block_avx512(mr, nr, kc, A_block, B_block,
                                                     C_block, ldc, alpha);
                        }
                    }
                }
            }
        }
    }
#else
    auto B_packed_storage = make_aligned_array<float>(b_buf_size_512);
    auto A_packed_storage = make_aligned_array<float>(a_buf_size);
    float* B_packed = B_packed_storage.get();
    float* A_packed = A_packed_storage.get();

    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(N - jc, NC);

        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(K - pc, KC);
            pack_b_panel_avx512(B + static_cast<std::size_t>(pc) * ldb + jc,
                                ldb, kc, nc, B_packed);

            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(M - ic, MC);
                pack_a_panel(A + static_cast<std::size_t>(ic) * lda + pc,
                             lda, mc, kc, A_packed);

                for (int jr = 0; jr < nc; jr += NR512) {
                    int nr = std::min(nc - jr, NR512);
                    const float* B_block = B_packed +
                        static_cast<std::size_t>(jr / NR512) * kc * NR512;
                    for (int ir = 0; ir < mc; ir += MR512) {
                        int mr = std::min(mc - ir, MR512);
                        const float* A_block = A_packed +
                            static_cast<std::size_t>(ir) * kc;
                        float* C_block = C + static_cast<std::size_t>(ic + ir) * ldc
                                           + jc + jr;
                        if (mr == MR512 && nr == NR512) {
                            inner_kernel_8x16_avx512(A_block, B_block, C_block, ldc, kc, alpha);
                        } else {
                            gemm_scalar_block_avx512(mr, nr, kc, A_block, B_block,
                                                     C_block, ldc, alpha);
                        }
                    }
                }
            }
        }
    }
#endif  // SIMD_ML_OPENMP
}

}  // namespace gemm
}  // namespace simd_ml
