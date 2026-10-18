/**
 * bf16_gemm.cpp — BF16-input GEMM with FP32 accumulation (AVX2)
 *
 * See bf16_gemm.hpp for the full design note. Key difference vs f16_gemm.cpp:
 *
 *   FP16 → FP32: requires F16C (_mm256_cvtph_ps)  — special ISA extension
 *   BF16 → FP32: zero-extend to 32 bits, left-shift 16 — pure AVX2 integers
 *
 * The zero-extend+shift trick works because BF16 is the top 16 bits of FP32:
 *   BF16[15:0] = FP32[31:16]  (sign + exponent + 7 MSBs of mantissa)
 * Placing those 16 bits in the upper half of a 32-bit word, with zeros in the
 * lower 16 bits, reconstructs a valid FP32 value with the lower mantissa bits
 * zeroed. This works for all normal values, subnormals, infinity, and NaN.
 *
 * ─── vdpbf16ps note ──────────────────────────────────────────────────────────
 * VDPBF16PS (Intel Ice Lake-SP+, AMD Zen4+) processes two BF16 pairs per cycle
 * into a FP32 accumulator. It requires panel-packed B in interleaved-pair
 * layout. This packing path is planned for ROADMAP.md v1.0 alongside NUMA-aware
 * allocation; it is NOT implemented here. The AVX2 zero-extend path is the
 * pragmatic choice for the batch sizes (1–128) this library targets in inference.
 *
 * bf16_avx512bf16_available() (dispatch.cpp) reports whether the CPU has
 * vdpbf16ps.
 */

#include "bf16_gemm.hpp"
#include "avx2_gemm_packed.hpp"

#include <algorithm>
#include <cstring>

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "bf16_gemm.cpp is an AVX2 unit: compile with -mavx2 -mfma (CMakeLists.txt)"
#endif
#include <immintrin.h>

namespace simd_ml {
namespace gemm {

static constexpr int kCols = 8;  // one YMM of output columns per step

// ─── BF16→FP32 conversion helpers (AVX2 integer, no F16C) ───────────────────

/**
 * bf16_to_fp32_scalar — convert one BF16 (uint16_t bit pattern) to float.
 *
 * BF16 occupies the upper 16 bits of a FP32 value. Place the BF16 bits
 * there (shift left 16) and reinterpret as float. Uses memcpy to avoid
 * strict-aliasing UB.
 */
static inline float bf16_to_fp32_scalar(uint16_t b) noexcept {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

/**
 * bf16_to_fp32_x8 — convert 8 consecutive BF16 values to 8 FP32 (__m256).
 *
 * Steps:
 *   _mm_loadu_si128         — load 8 × uint16_t (128 bits)
 *   _mm256_cvtepu16_epi32   — zero-extend each uint16_t to uint32_t (AVX2)
 *   _mm256_slli_epi32(., 16) — shift left 16: BF16 bits → FP32 upper half
 *   _mm256_castsi256_ps     — reinterpret the integer bits as float (no-op)
 *
 * No division, no special ISA. 3 instructions for 8 values, all integer.
 */
static inline __m256 bf16_to_fp32_x8(const uint16_t* ptr) noexcept {
    __m128i v16      = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    __m256i v32      = _mm256_cvtepu16_epi32(v16);
    __m256i fp32bits = _mm256_slli_epi32(v32, 16);
    return _mm256_castsi256_ps(fp32bits);
}


// ─── sgemm_bf16_avx2 implementation ──────────────────────────────────────────
void sgemm_bf16_avx2(int M, int N, int K,
                     float alpha,
                     const uint16_t* __restrict__ A, int lda,
                     const uint16_t* __restrict__ B, int ldb,
                     float beta,
                     float*          __restrict__ C, int ldc) noexcept {
    // K == 0 is valid (BLAS semantics): A·B is empty, so C = beta·C.
    // The beta pass below runs and the K-loops do nothing.
    if (M <= 0 || N <= 0 || K < 0) return;

    // ── Beta pre-pass: C ← beta * C ──────────────────────────────────────────
    // beta=0 must zero C explicitly — never multiply existing values
    // (which may be NaN/inf from uninitialised output buffers).
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
    // Inner kernel (vectorised, kCols=8 columns per iteration):
    //   acc[0..7] = Σ_p  fp32(A[i][p]) * fp32(B[p][j..j+7])
    //
    // BF16→FP32 conversions:
    //   A[i][p]:      bf16_to_fp32_scalar → scalar broadcast → __m256
    //   B[p][j..j+7]: bf16_to_fp32_x8    → __m256
    //
    // B is accessed with stride ldb uint16_t elements between k-steps.
    // For small matrices (max dim ≤ 128) the entire B fits in L2 cache.

    const __m256 alpha_v = _mm256_set1_ps(alpha);

    for (int i = 0; i < M; ++i) {
        const uint16_t* a_row = A + static_cast<std::size_t>(i) * lda;
        float*          c_row = C + static_cast<std::size_t>(i) * ldc;

        // ── Vectorised kCols=8 columns per iteration ─────────────────────────────
        int j = 0;
        for (; j + kCols <= N; j += kCols) {
            __m256 acc = _mm256_setzero_ps();
            for (int p = 0; p < K; ++p) {
                // Convert BF16 A[i][p] → FP32 scalar, broadcast to all 8 lanes.
                // Scalar conversion (3 cycles): shift + memcpy + set1.
                __m256 a_bcast = _mm256_set1_ps(bf16_to_fp32_scalar(a_row[p]));

                // Convert 8 BF16 B[p][j..j+7] → FP32 vector (2 AVX2 ops):
                // cvtepu16_epi32 zero-extends, slli shifts them into FP32 position.
                __m256 b_vec = bf16_to_fp32_x8(
                    B + static_cast<std::size_t>(p) * ldb + j);

                acc = _mm256_fmadd_ps(a_bcast, b_vec, acc);
            }

            if (alpha != 1.0f) acc = _mm256_mul_ps(acc, alpha_v);

            // C[i][j..j+7] += alpha * acc
            __m256 cv = _mm256_loadu_ps(c_row + j);
            _mm256_storeu_ps(c_row + j, _mm256_add_ps(cv, acc));
        }

        // ── Scalar tail for remaining N % kCols columns ──────────────────────────
        for (; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                sum += bf16_to_fp32_scalar(a_row[p]) *
                       bf16_to_fp32_scalar(
                           B[static_cast<std::size_t>(p) * ldb + j]);
            }
            c_row[j] += alpha * sum;
        }
    }
}

}  // namespace gemm
}  // namespace simd_ml
