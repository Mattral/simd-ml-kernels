#pragma once

/**
 * q8_0_rows.hpp — the Q8_0 GEMV inner loop on YMM registers, shared by the
 * AVX2 (vpmaddubsw), AVX-VNNI and AVX512_VNNI units, which differ only in the
 * 32-byte integer dot (Dot::dot). Everything is in an anonymous namespace:
 * each unit compiles its own copy with its own flags (tests/check_isa_units.py).
 *
 * Integer dot of one block, s8 × s8: vpdpbusd and vpmaddubsw take one
 * unsigned and one signed operand, so the sign of x moves onto w (ggml's
 * trick, mul_sum_i8_pairs): |x| · sign(w, x) = x · w per byte (vpabsb,
 * vpsignb). Both operands are in [-127, 127] (Q8_0 never stores -128), so
 * |x| fits a byte and vpsignb never negates -128; vpmaddubsw's int16 pair
 * sums are at most 2·127·127 = 32258 < 32767, so they do not saturate.
 * (sgemm_u8s8 subtracts a zero point instead, with column sums; here both
 * operands are signed and there is no zero point, so the sign trick costs one
 * vpsignb per 32 weights and nothing per block.)
 *
 * Each block (32 weights, 34 bytes) of R rows: one unaligned 32-byte load of
 * the quants, vpsignb, the dot (8 int32 lanes), vcvtdq2ps, the scale
 * fp16(dw)·dx broadcast (vpbroadcastw + vcvtph2ps, one vmulps) and one FMA
 * into the row's 8-lane accumulator; x's block (|x|, x, dx) is loaded once
 * per R rows. R rows are walked together along K: R sequential streams for
 * the prefetchers, as the FP32 GEMV (DESIGN.md §2.7).
 */

#include "q8_0.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace simd_ml {
namespace quant {
namespace {

#if defined(__GNUC__) || defined(__clang__)
#  define SIMD_ML_Q8_INLINE inline __attribute__((always_inline))
#else
#  define SIMD_ML_Q8_INLINE inline
#endif

inline float hsum8(__m256 v) noexcept {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

template <class Dot, int R>
SIMD_ML_Q8_INLINE void q8_rows_block(int nb, const std::int8_t* __restrict__ xq, const float* __restrict__ dx,
                                     const std::uint8_t* __restrict__ W, std::size_t row_bytes,
                                     float* __restrict__ y) noexcept {
    __m256 acc[R];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + kQ8_0Values * b));
        const __m256i ax = _mm256_abs_epi8(xv);
        const __m256 dxv = _mm256_broadcast_ss(dx + b);
        const std::uint8_t* blk = W + static_cast<std::size_t>(b) * kQ8_0Bytes;
        for (int r = 0; r < R; ++r) {
            const std::uint8_t* p = blk + r * row_bytes;
            // The same row of the next R-row group, one line per two blocks:
            // its streams start warm. Dev runs (8 rows): 67-91% of the read
            // roof without, 90-100% with; 512 or 1024 bytes ahead in the same
            // row changed nothing. Past the end of W it only touches a line
            // (prefetches do not fault).
            if ((b & 1) == 0) _mm_prefetch(reinterpret_cast<const char*>(p + R * row_bytes), _MM_HINT_T0);
            const __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 2));
            const __m256 dot = _mm256_cvtepi32_ps(Dot::dot(ax, _mm256_sign_epi8(wv, xv)));
            std::uint16_t h;
            std::memcpy(&h, p, 2);
            const __m256 s = _mm256_mul_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(h))), dxv);
            acc[r] = _mm256_fmadd_ps(dot, s, acc[r]);
        }
    }
    for (int r = 0; r < R; ++r) y[r] = hsum8(acc[r]);
}

/** The last rem < R rows: the block kernel for exactly rem rows. */
template <class Dot, int R>
SIMD_ML_Q8_INLINE void q8_rows_tail(int rem, int nb, const std::int8_t* xq, const float* dx,
                                    const std::uint8_t* W, std::size_t row_bytes, float* y) noexcept {
    if constexpr (R > 0) {
        if (rem == R) q8_rows_block<Dot, R>(nb, xq, dx, W, row_bytes, y);
        else q8_rows_tail<Dot, R - 1>(rem, nb, xq, dx, W, row_bytes, y);
    }
}

/** y[0..N) for Q8_0 rows Wq, x quantized to (xq, dx); kRows rows at a time. */
template <class Dot, int kRows>
SIMD_ML_Q8_INLINE void q8_rows(int N, int nb, const std::int8_t* xq, const float* dx,
                               const std::uint8_t* Wq, float* y) noexcept {
    const std::size_t row_bytes = static_cast<std::size_t>(nb) * kQ8_0Bytes;
    int n = 0;
    for (; n + kRows <= N; n += kRows)
        q8_rows_block<Dot, kRows>(nb, xq, dx, Wq + n * row_bytes, row_bytes, y + n);
    q8_rows_tail<Dot, kRows - 1>(N - n, nb, xq, dx, Wq + n * row_bytes, row_bytes, y + n);
}

}  // namespace
}  // namespace quant
}  // namespace simd_ml
