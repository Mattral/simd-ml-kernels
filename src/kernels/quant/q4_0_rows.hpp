#pragma once

/**
 * q4_0_rows.hpp — the Q4_0 GEMV inner loop on YMM registers, shared by the
 * AVX2 (vpmaddubsw), AVX-VNNI and AVX512_VNNI units, which differ only in the
 * 32-byte integer dot-accumulate (Dot::dot_acc). Everything is in an
 * anonymous namespace: each unit compiles its own copy with its own flags
 * (tests/check_isa_units.py).
 *
 * Why two rows per register. A Q4_0 block is 18 bytes, so at the read roof
 * (about 30 GB/s on one core of the reference laptop) a (row, block) arrives
 * every ~0.6 ns, while the Q8_0 loop's 9 vector uops per (row, block) take
 * ~3 cycles (~1 ns; measured with the weights in L2). The loop here holds
 * rows r and r + 1 in one YMM register, lanes 0-3 and 4-7, so the
 * conversion, the scale and the FMA are shared by two rows:
 *
 *   per row pair and block (6.5 vector instructions per row and block):
 *     vmovdqu xmm + vinserti128   the two rows' 16 quant bytes, one per lane
 *     vpand, vpsrlw + vpand       low nibbles (values 0-15), high (16-31)
 *     2 vpdpbusd                  low against x[0..15], high against x[16..31]
 *                                 (x's halves broadcast to both lanes)
 *     vmovd + vpinsrd + vpshufb   the two fp16 scales, each 4 times
 *     vcvtph2ps, vmulps (dx)      8 FP32 scales; vcvtdq2ps; one FMA
 *
 * Nibbles: q is unsigned in [0, 15] and is vpdpbusd's unsigned operand, x's
 * Q8 quants the signed one, so Σ (q − 8)·x = Σ q·x − 8·Σ x needs no sign
 * trick (ggml subtracts 8 from q, then vpsignb / vpabsb as for Q8_0). The
 * −8·Σ x part depends on x only: computed once per block for all rows (two
 * dots of 8 with x's halves, negated) and used as the dot's starting
 * accumulator, so with VNNI it costs nothing per row. vpmaddubsw's int16
 * pair sums are at most 2·15·127 = 3810: no saturation.
 *
 * An odd last row uses the one-row loop (q4_row_single): 32 values in one
 * register, vbroadcasti128 + vpsrlvd (upper lane's dwords right by 4) +
 * vpand for the nibbles, one vpdpbusd.
 */

#include "q4_0.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace simd_ml {
namespace quant {
namespace {

#if defined(__GNUC__) || defined(__clang__)
#  define SIMD_ML_Q4_INLINE inline __attribute__((always_inline))
#else
#  define SIMD_ML_Q4_INLINE inline
#endif

inline float q4_hsum4(__m128 s) noexcept {
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline std::int32_t q4_load32(const std::uint8_t* p) noexcept {
    std::int32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

/**
 * 2P rows of Q4_0 (P row pairs) against x's nb Q8 blocks. PF: software
 * prefetch of the same row of the next 2P-row group every PF blocks (0: none).
 */
template <class Dot, int P, int PF>
SIMD_ML_Q4_INLINE void q4_pairs_block(int nb, const std::int8_t* __restrict__ xq, const float* __restrict__ dx,
                                      const std::uint8_t* __restrict__ W, std::size_t row_bytes,
                                      float* __restrict__ y) noexcept {
    const __m256i low4 = _mm256_set1_epi8(0x0f);
    const __m256i eight = _mm256_set1_epi8(8);
    // Bytes 0-1 (row r's scale) four times, then bytes 4-5 (row r + 1's).
    const __m128i scale_bytes = _mm_setr_epi8(0, 1, 0, 1, 0, 1, 0, 1, 4, 5, 4, 5, 4, 5, 4, 5);
    __m256 acc[P];
    for (int i = 0; i < P; ++i) acc[i] = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const std::int8_t* xb = xq + kQ4_0Values * b;
        const __m256i xlo = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(xb)));
        const __m256i xhi = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(xb + 16)));
        // −8·(Σ of x's 4 low-half and 4 high-half bytes) per int32 lane.
        const __m256i bias = _mm256_sub_epi32(
            _mm256_setzero_si256(), Dot::dot_acc(Dot::dot_acc(_mm256_setzero_si256(), eight, xlo), eight, xhi));
        const __m256 dxv = _mm256_broadcast_ss(dx + b);
        const std::uint8_t* blk = W + static_cast<std::size_t>(b) * kQ4_0Bytes;
        for (int i = 0; i < P; ++i) {
            const std::uint8_t* p0 = blk + (2 * i) * row_bytes;
            const std::uint8_t* p1 = p0 + row_bytes;
            if constexpr (PF > 0) {
                if (b % PF == 0) {
                    _mm_prefetch(reinterpret_cast<const char*>(p0 + 2 * P * row_bytes), _MM_HINT_T0);
                    _mm_prefetch(reinterpret_cast<const char*>(p1 + 2 * P * row_bytes), _MM_HINT_T0);
                }
            }
            const __m256i v = _mm256_inserti128_si256(
                _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p0 + 2))),
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(p1 + 2)), 1);
            const __m256i lo = _mm256_and_si256(v, low4);
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low4);
            const __m256i d = Dot::dot_acc(Dot::dot_acc(bias, lo, xlo), hi, xhi);
            const __m128i h = _mm_shuffle_epi8(_mm_insert_epi32(_mm_cvtsi32_si128(q4_load32(p0)), q4_load32(p1), 1),
                                               scale_bytes);
            const __m256 s = _mm256_mul_ps(_mm256_cvtph_ps(h), dxv);
            acc[i] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), s, acc[i]);
        }
    }
    for (int i = 0; i < P; ++i) {
        y[2 * i] = q4_hsum4(_mm256_castps256_ps128(acc[i]));
        y[2 * i + 1] = q4_hsum4(_mm256_extractf128_ps(acc[i], 1));
    }
}

/** One row (the odd last one): all 32 values of a block in one register. */
template <class Dot>
SIMD_ML_Q4_INLINE void q4_row_single(int nb, const std::int8_t* __restrict__ xq, const float* __restrict__ dx,
                                     const std::uint8_t* __restrict__ W, float* __restrict__ y) noexcept {
    const __m256i shift = _mm256_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4);
    const __m256i low4 = _mm256_set1_epi8(0x0f);
    const __m256i eight = _mm256_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const std::uint8_t* p = W + static_cast<std::size_t>(b) * kQ4_0Bytes;
        const __m256i xv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xq + kQ4_0Values * b));
        const __m256i bias = _mm256_sub_epi32(_mm256_setzero_si256(),
                                              Dot::dot_acc(_mm256_setzero_si256(), eight, xv));
        const __m256i raw = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 2)));
        const __m256i qv = _mm256_and_si256(_mm256_srlv_epi32(raw, shift), low4);
        std::uint16_t h;
        std::memcpy(&h, p, 2);
        const __m256 s = _mm256_mul_ps(_mm256_cvtph_ps(_mm_set1_epi16(static_cast<short>(h))),
                                       _mm256_broadcast_ss(dx + b));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(Dot::dot_acc(bias, qv, xv)), s, acc);
    }
    *y = q4_hsum4(_mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1)));
}

/** The last rem < 2P rows: rem / 2 pairs, then a single row if rem is odd. */
template <class Dot, int P, int PF>
SIMD_ML_Q4_INLINE void q4_rows_tail(int rem, int nb, const std::int8_t* xq, const float* dx,
                                    const std::uint8_t* W, std::size_t row_bytes, float* y) noexcept {
    if constexpr (P > 0) {
        if (rem / 2 == P) {
            q4_pairs_block<Dot, P, PF>(nb, xq, dx, W, row_bytes, y);
            if (rem & 1) q4_row_single<Dot>(nb, xq, dx, W + 2 * P * row_bytes, y + 2 * P);
        } else {
            q4_rows_tail<Dot, P - 1, PF>(rem, nb, xq, dx, W, row_bytes, y);
        }
    } else if (rem == 1) {
        q4_row_single<Dot>(nb, xq, dx, W, y);
    }
}

/** y[0..N) for Q4_0 rows Wq, x quantized to (xq, dx); kRows (even) rows at a time. */
template <class Dot, int kRows, int PF>
SIMD_ML_Q4_INLINE void q4_rows(int N, int nb, const std::int8_t* xq, const float* dx,
                               const std::uint8_t* Wq, float* y) noexcept {
    static_assert(kRows % 2 == 0, "rows are walked in pairs");
    const std::size_t row_bytes = static_cast<std::size_t>(nb) * kQ4_0Bytes;
    int n = 0;
    for (; n + kRows <= N; n += kRows)
        q4_pairs_block<Dot, kRows / 2, PF>(nb, xq, dx, Wq + n * row_bytes, row_bytes, y + n);
    q4_rows_tail<Dot, kRows / 2 - 1, PF>(N - n, nb, xq, dx, Wq + n * row_bytes, row_bytes, y + n);
}

}  // namespace
}  // namespace quant
}  // namespace simd_ml
