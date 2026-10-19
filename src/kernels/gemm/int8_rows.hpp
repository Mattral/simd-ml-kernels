#pragma once

/**
 * int8_rows.hpp — sgemm_u8s8 for one row of A with B stored [N, K]
 * (trans_b, M == 1: x·Wᵀ for one token), reading A and B in place. The
 * packed path (avxvnni_gemm.cpp etc.) copies all of B into micro-panels on
 * every call, which costs as much as using it once: at 1×4096×4096 it ran at
 * 0.46× PyTorch on AVX-VNNI (DESIGN.md §4.2). Here R rows of B are walked
 * together along K (R sequential streams, as the FP32 GEMV, §2.7) and A's
 * 32-byte step is shared by the R rows.
 *
 *   C[j] = a_scale · b_scales[j] · (Σ_k A[k]·B[j][k] − zp·Σ_k B[j][k])
 *
 * Step policies (32 values of k per step):
 *  - Vnni: vpdpbusd(acc, a, b) for Σ a·b (u8 × s8, the instruction's own
 *    operand order) and vpdpbusd(acc, ones, b) for Σ b;
 *  - Int16 (AVX2 without VNNI): a − zp and b widened to int16 (exact,
 *    |a − zp| <= 255), vpmaddwd. vpmaddubsw cannot be used: 2·255·128
 *    overflows its int16 pair sums.
 * Lanes are int32: one step adds at most 2·2·255·128 = 130560 to a lane, so
 * K is taken in chunks of kChunk = 2^18 (8192 steps, < 2^31 / 130560 =
 * 16448), and each chunk's lanes are summed into int64. The total is exact;
 * the only roundings are the conversion to float and the two scale products,
 * well inside the bound of int8_gemm.hpp. A K tail (K % 32) is copied into
 * zeroed 32-byte buffers, so nothing past a row is read.
 *
 * Anonymous namespace: each unit compiles its own copy (tests/check_isa_units.py).
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace simd_ml {
namespace gemm {
namespace {

#if defined(__GNUC__) || defined(__clang__)
#  define SIMD_ML_ROWS_INLINE inline __attribute__((always_inline))
#else
#  define SIMD_ML_ROWS_INLINE inline
#endif

constexpr int kRowStep = 32;
constexpr int kChunk = 1 << 18;

inline long long hsum_epi32_to_i64(__m256i v) noexcept {
    // Sign-extend the 8 lanes to int64 before adding: 8 lanes of up to 2^31 overflow int32.
    const __m256i lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v));
    const __m256i hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(v, 1));
    const __m256i s = _mm256_add_epi64(lo, hi);
    const __m128i t = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    return _mm_cvtsi128_si64(t) + _mm_extract_epi64(t, 1);
}

template <class Step, int R>
SIMD_ML_ROWS_INLINE void u8s8_rows_block(int K, const std::uint8_t* __restrict__ a, int zp,
                                         const std::int8_t* __restrict__ B, std::ptrdiff_t ldb,
                                         float a_scale, const float* __restrict__ scales,
                                         float* __restrict__ c) noexcept {
    long long total[R];
    for (int r = 0; r < R; ++r) total[r] = 0;
    const typename Step::Zp z = Step::zero_point(zp);
    for (int k0 = 0; k0 < K; k0 += kChunk) {
        const int k1 = K - k0 < kChunk ? K : k0 + kChunk;
        __m256i ab[R], sb[R];
        for (int r = 0; r < R; ++r) ab[r] = sb[r] = _mm256_setzero_si256();
        int k = k0;
        for (; k + kRowStep <= k1; k += kRowStep) {
            const typename Step::A av = Step::load_a(a + k, z);
            for (int r = 0; r < R; ++r) {
                // The next R-row group's same row, one line per two steps (as
                // the Q8_0 GEMV, q8_0_rows.hpp); prefetches do not fault.
                if ((k & 63) == 0) _mm_prefetch(reinterpret_cast<const char*>(B + (r + R) * ldb + k), _MM_HINT_T0);
                Step::step(av, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(B + r * ldb + k)), ab[r], sb[r]);
            }
        }
        if (k < k1) {   // K tail (last chunk only): zero-padded copies
            alignas(32) std::uint8_t at[kRowStep] = {};
            std::memcpy(at, a + k, static_cast<std::size_t>(k1 - k));
            const typename Step::A av = Step::load_a(at, z);
            for (int r = 0; r < R; ++r) {
                alignas(32) std::int8_t bt[kRowStep] = {};
                std::memcpy(bt, B + r * ldb + k, static_cast<std::size_t>(k1 - k));
                Step::step(av, _mm256_load_si256(reinterpret_cast<const __m256i*>(bt)), ab[r], sb[r]);
            }
        }
        for (int r = 0; r < R; ++r) total[r] += Step::finish(ab[r], sb[r], zp);
    }
    for (int r = 0; r < R; ++r) c[r] = static_cast<float>(total[r]) * (a_scale * scales[r]);
}

template <class Step, int R>
SIMD_ML_ROWS_INLINE void u8s8_rows_tail(int rem, int K, const std::uint8_t* a, int zp, const std::int8_t* B,
                                        std::ptrdiff_t ldb, float a_scale, const float* scales,
                                        float* c) noexcept {
    if constexpr (R > 0) {
        if (rem == R) u8s8_rows_block<Step, R>(K, a, zp, B, ldb, a_scale, scales, c);
        else u8s8_rows_tail<Step, R - 1>(rem, K, a, zp, B, ldb, a_scale, scales, c);
    }
}

/** C[0..N) = one row a [K] times B [N, K]ᵀ, kRows rows of B at a time. */
template <class Step, int kRows>
SIMD_ML_ROWS_INLINE void u8s8_rows(int N, int K, const std::uint8_t* a, int zp, const std::int8_t* B,
                                   int ldb, float a_scale, const float* scales, float* C) noexcept {
    int j = 0;
    for (; j + kRows <= N; j += kRows)
        u8s8_rows_block<Step, kRows>(K, a, zp, B + static_cast<std::ptrdiff_t>(j) * ldb, ldb, a_scale,
                                     scales + j, C + j);
    u8s8_rows_tail<Step, kRows - 1>(N - j, K, a, zp, B + static_cast<std::ptrdiff_t>(j) * ldb, ldb, a_scale,
                                    scales + j, C + j);
}

}  // namespace
}  // namespace gemm
}  // namespace simd_ml
