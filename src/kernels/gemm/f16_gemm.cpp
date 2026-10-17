/**
 * f16_gemm.cpp — FP16-input GEMM with FP32 accumulation (F16C + AVX2)
 *
 * See f16_gemm.hpp for the full design note. Summary:
 *   C[fp32] = alpha * A[fp16] @ B[fp16] + beta * C[fp32]
 *
 * This file is compiled with -mf16c (set via CMake COMPILE_OPTIONS so the
 * flag is present even on machines where -march=native might not include it).
 * All F16C intrinsics are guarded by #ifdef __F16C__ so the file compiles to
 * a safe no-op stub on compilers / CPUs that lack the extension.
 */

#include "f16_gemm.hpp"
#include "avx2_gemm_packed.hpp"  // for NR = 8 constant
#include "../../dispatch/cpuid.hpp"

#include <algorithm>
#include <cstring>

#ifdef __F16C__
#  include <immintrin.h>
#endif

namespace simd_ml {
namespace gemm {

// ─── Runtime capability check ─────────────────────────────────────────────────
bool f16c_available() noexcept {
#ifdef __F16C__
    // CPUID + XGETBV via CpuFeatures rather than __builtin_cpu_supports("f16c"),
    // which Clang <= 15 rejects as an unknown feature string.
    static const bool available = dispatch::CpuFeatures::detect().has_f16c;
    return available;
#else
    return false;
#endif
}

// ─── sgemm_f16_avx2 implementation ───────────────────────────────────────────
void sgemm_f16_avx2(int M, int N, int K,
                    float alpha,
                    const uint16_t* __restrict__ A, int lda,
                    const uint16_t* __restrict__ B, int ldb,
                    float beta,
                    float*          __restrict__ C, int ldc) noexcept {
#ifndef __F16C__
    // Compiled without F16C. The Python binding checks f16c_available()
    // before calling here, so this branch is unreachable in correct use.
    (void)M; (void)N; (void)K; (void)alpha;
    (void)A; (void)lda; (void)B; (void)ldb;
    (void)beta; (void)C; (void)ldc;
    return;
#else
    if (M <= 0 || N <= 0 || K <= 0) return;

    // ── Beta pre-pass: C ← beta * C ──────────────────────────────────────────
    // Mirrors scale_matrix_c in sgemm_packed: beta=0 must zero C explicitly
    // (never multiply in case C contains NaN from uninitialised memory).
    if (beta != 1.0f) {
        for (int i = 0; i < M; ++i) {
            float* crow = C + static_cast<std::size_t>(i) * ldc;
            if (beta == 0.0f) {
                std::fill(crow, crow + N, 0.0f);
            } else {
                for (int j = 0; j < N; ++j) crow[j] *= beta;
            }
        }
    }

    // ── Main loop: row i of A × all columns of B → row i of C ───────────────
    //
    // For each output row i:
    //   for j in [0, N) step 8:
    //     acc[0..7] = Σ_p  fp32(A[i][p]) * fp32(B[p][j..j+7])
    //     C[i][j..j+7] += alpha * acc
    //
    // F16C conversions:
    //   A[i][p]:  scalar: _cvtsh_ss(a_row[p])   → float
    //   B[p][j..j+7]: _mm256_cvtph_ps(_mm_loadu_si128(...)) → __m256
    //
    // B is accessed with stride ldb uint16_t elements between k-steps.
    // For small matrices (all dims ≤ 128) B fits in L2; hardware prefetch
    // handles the stride. For larger matrices, consider a packing path
    // (not implemented here — see ROADMAP.md v0.9/v1.0).

    const __m256 alpha_v = _mm256_set1_ps(alpha);

    for (int i = 0; i < M; ++i) {
        const uint16_t* a_row = A + static_cast<std::size_t>(i) * lda;
        float*          c_row = C + static_cast<std::size_t>(i) * ldc;

        // ── Vectorised NR=8 columns per iteration ─────────────────────────────
        int j = 0;
        for (; j + NR <= N; j += NR) {
            __m256 acc = _mm256_setzero_ps();
            for (int p = 0; p < K; ++p) {
                // Convert A[i][p] from FP16 → FP32 and broadcast to all 8 lanes.
                // _cvtsh_ss uses the F16C scalar conversion instruction (VCVTPH2PS
                // with a single element); the broadcast _mm256_set1_ps is free
                // after it (or folded into a subsequent vbroadcastss by the compiler).
                __m256 a_bcast = _mm256_set1_ps(_cvtsh_ss(a_row[p]));

                // Load B[p][j..j+7] (8 × FP16 = 128 bits = __m128i) and convert
                // to 8 × FP32 (__m256) in a single VCVTPH2PS instruction.
                __m256 b_vec = _mm256_cvtph_ps(
                    _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(
                            B + static_cast<std::size_t>(p) * ldb + j)));

                acc = _mm256_fmadd_ps(a_bcast, b_vec, acc);
            }

            // Apply alpha scale (branch-free when alpha == 1.0f)
            if (alpha != 1.0f) acc = _mm256_mul_ps(acc, alpha_v);

            // Accumulate into C: C[i][j..j+7] += alpha * acc
            __m256 cv = _mm256_loadu_ps(c_row + j);
            _mm256_storeu_ps(c_row + j, _mm256_add_ps(cv, acc));
        }

        // ── Scalar tail for N % NR remaining columns ──────────────────────────
        for (; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                // Scalar F16→F32 for both A and B (tail is rare, code size wins)
                sum += _cvtsh_ss(a_row[p]) *
                       _cvtsh_ss(B[static_cast<std::size_t>(p) * ldb + j]);
            }
            c_row[j] += alpha * sum;
        }
    }
#endif  // __F16C__
}

}  // namespace gemm
}  // namespace simd_ml
