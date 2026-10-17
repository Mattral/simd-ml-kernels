/**
 * avx_matmul_avx512.cpp — the 6×32 AVX-512 micro-kernel of simd_sgemm
 * (avx_matmul.cpp). An AVX-512 unit; avx_matmul.cpp calls it only when
 * kernels_for(Isa::avx512) is non-null.
 */

#include <immintrin.h>

#if !defined(__AVX512F__)
#  error "avx_matmul_avx512.cpp is an AVX-512 unit: compile with -mavx512f (CMakeLists.txt)"
#endif

static constexpr int MR = 6;   // must match avx_matmul.cpp

/**
 * gemm_micro_6x32_avx512 — 6×32 blocking using dual 512-bit ZMM accumulators.
 *
 * Doubles the N-width vs the AVX2 6×16 kernel using AVX-512's wider 512-bit
 * registers AND its much larger physical register file (32 ZMM vs 16 YMM),
 * so there is no register-pressure trade-off here the way there was choosing
 * between the AVX2 8×8 and 6×16 layouts.
 *
 * ─── Register budget (32 ZMM available) ──────────────────────────────────────
 *   12 ZMM  — accumulators: 6 rows × 2 halves (lo=cols[0:16), hi=cols[16:32))
 *    2 ZMM  — B panel (b_lo, b_hi) reloaded each k-step
 *    1 ZMM  — A broadcast (reused per row per k-step)
 *   15 ZMM total — 17 registers of headroom vs. the 32 available; zero
 *   spill risk even with aggressive unrolling.
 *
 * ─── Correctness note (see ROADMAP.md §v0.8 history) ────────────────────────
 * An earlier version of this kernel declared NR512=32 but only ever loaded
 * and accumulated ONE __m512 (16 floats) per row — silently leaving the
 * upper 16 columns of every 32-wide output tile untouched. That bug is why
 * AVX-512 dispatch was disabled for an entire release cycle. This version
 * is proven correct via tests/test_gemm.cpp's
 * test_gemm_avx512_full_coverage(), which forces 100% of the output
 * through this exact kernel (M and N chosen so no edge/tail block can
 * mask a broken full-block path) and independently checks both
 * accumulator halves against a scalar reference.
 */
void gemm_micro_6x32_avx512(int kc,
                                    const float* __restrict__ A_panel,
                                    const float* __restrict__ B_panel,
                                    float*       __restrict__ C,
                                    int ldc) noexcept {
    // Load existing C values — both halves of all 6 rows (accumulate, not overwrite)
    __m512 c0_lo = _mm512_loadu_ps(C + 0*ldc +  0), c0_hi = _mm512_loadu_ps(C + 0*ldc + 16);
    __m512 c1_lo = _mm512_loadu_ps(C + 1*ldc +  0), c1_hi = _mm512_loadu_ps(C + 1*ldc + 16);
    __m512 c2_lo = _mm512_loadu_ps(C + 2*ldc +  0), c2_hi = _mm512_loadu_ps(C + 2*ldc + 16);
    __m512 c3_lo = _mm512_loadu_ps(C + 3*ldc +  0), c3_hi = _mm512_loadu_ps(C + 3*ldc + 16);
    __m512 c4_lo = _mm512_loadu_ps(C + 4*ldc +  0), c4_hi = _mm512_loadu_ps(C + 4*ldc + 16);
    __m512 c5_lo = _mm512_loadu_ps(C + 5*ldc +  0), c5_hi = _mm512_loadu_ps(C + 5*ldc + 16);

    const float* ap = A_panel;
    const float* bp = B_panel;

    for (int k = 0; k < kc; ++k, ap += MR, bp += 32) {
        // Load BOTH halves of the 32-wide B panel row for this k-step.
        __m512 b_lo = _mm512_loadu_ps(bp);       // cols [0, 16)
        __m512 b_hi = _mm512_loadu_ps(bp + 16);  // cols [16, 32)

        __m512 a0 = _mm512_set1_ps(ap[0]);
        c0_lo = _mm512_fmadd_ps(a0, b_lo, c0_lo);
        c0_hi = _mm512_fmadd_ps(a0, b_hi, c0_hi);

        __m512 a1 = _mm512_set1_ps(ap[1]);
        c1_lo = _mm512_fmadd_ps(a1, b_lo, c1_lo);
        c1_hi = _mm512_fmadd_ps(a1, b_hi, c1_hi);

        __m512 a2 = _mm512_set1_ps(ap[2]);
        c2_lo = _mm512_fmadd_ps(a2, b_lo, c2_lo);
        c2_hi = _mm512_fmadd_ps(a2, b_hi, c2_hi);

        __m512 a3 = _mm512_set1_ps(ap[3]);
        c3_lo = _mm512_fmadd_ps(a3, b_lo, c3_lo);
        c3_hi = _mm512_fmadd_ps(a3, b_hi, c3_hi);

        __m512 a4 = _mm512_set1_ps(ap[4]);
        c4_lo = _mm512_fmadd_ps(a4, b_lo, c4_lo);
        c4_hi = _mm512_fmadd_ps(a4, b_hi, c4_hi);

        __m512 a5 = _mm512_set1_ps(ap[5]);
        c5_lo = _mm512_fmadd_ps(a5, b_lo, c5_lo);
        c5_hi = _mm512_fmadd_ps(a5, b_hi, c5_hi);
    }

    // Write back BOTH halves of every row — this is the exact line the
    // original buggy version omitted for the "hi" half.
    _mm512_storeu_ps(C + 0*ldc +  0, c0_lo);  _mm512_storeu_ps(C + 0*ldc + 16, c0_hi);
    _mm512_storeu_ps(C + 1*ldc +  0, c1_lo);  _mm512_storeu_ps(C + 1*ldc + 16, c1_hi);
    _mm512_storeu_ps(C + 2*ldc +  0, c2_lo);  _mm512_storeu_ps(C + 2*ldc + 16, c2_hi);
    _mm512_storeu_ps(C + 3*ldc +  0, c3_lo);  _mm512_storeu_ps(C + 3*ldc + 16, c3_hi);
    _mm512_storeu_ps(C + 4*ldc +  0, c4_lo);  _mm512_storeu_ps(C + 4*ldc + 16, c4_hi);
    _mm512_storeu_ps(C + 5*ldc +  0, c5_lo);  _mm512_storeu_ps(C + 5*ldc + 16, c5_hi);
}
