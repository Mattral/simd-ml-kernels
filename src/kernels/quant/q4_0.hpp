#pragma once

/**
 * q4_0.hpp — ggml's Q4_0 weight format and the decode GEMV on it
 * (docs/DESIGN.md §2.9).
 *
 * Format (byte-identical to ggml's block_q4_0): a row of K values is K/32
 * blocks of 18 bytes, each an IEEE binary16 scale d (little-endian) followed
 * by 16 bytes of 4-bit quants: byte j holds element j in its low nibble and
 * element 16 + j in its high nibble. Value i of the block is d·(q[i] − 8),
 * q in [0, 15]. A matrix W [N, K] is its N rows one after the other,
 * N·(K/32)·18 bytes. Quantization is ggml's quantize_row_q4_0_ref: per block
 * max = the value of largest magnitude (the first one on a tie), d = max / −8
 * (FP32), q = min(15, trunc(w·(1/d) + 8.5)) (each step rounded in FP32;
 * id = 0 if d = 0), and the stored scale is d rounded to binary16 (round to
 * nearest even). The largest-magnitude value always maps to q = 0, so it is
 * stored as −8·d = max (up to the scale's binary16 rounding); the other end
 * of the range, +7·d, is never reached by it. One case differs: a block whose
 * largest magnitude is below 8/FLT_MAX (1/d overflows) stores q = 8 (value
 * 0), where the C reference converts ±inf or NaN to int8 (undefined); its
 * scale is binary16 zero either way.
 *
 *   gemv_q4_0:  y[n] = Σ_b fp16(dw[n][b]) · dx[b] · Σ_i (qw[n][b][i] − 8) · qx[b][i]
 *
 * x is quantized once per call to Q8 blocks (q8_0.hpp: quantize_x_q8, FP32
 * scales), as ggml's vec_dot_q4_0_q8_0 does. The 32-term integer sums are
 * exact (|Σ| <= 32·8·127). Error bound of every path, against that sum taken
 * exactly (float64):
 *
 *   |y − y64| <= 2·(K/32 + 4)·u·Σ_b |fp16(dw_b)·dx_b|·Σ_i |(qw − 8)·qx|,  u = 2^-24.
 *
 * x with a NaN or ±Inf: every y is NaN. Scales in Wq that are not finite
 * (only possible in bytes not made by quantize_q4_0_rows): unspecified.
 */

#include "q8_0.hpp"

#include <cstddef>
#include <cstdint>

namespace simd_ml {
namespace quant {

constexpr int kQ4_0Values = 32;   ///< values per block
constexpr int kQ4_0Bytes  = 18;   ///< bytes per block (binary16 scale + 16 bytes of nibbles)

/** Bytes of one Q4_0 row of K values (K % 32 == 0). */
constexpr std::size_t q4_0_row_bytes(int K) {
    return static_cast<std::size_t>(K / kQ4_0Values) * kQ4_0Bytes;
}

// ─── Scalar unit (q4_0_scalar.cpp; no -ffast-math, no FP contraction) ───────

/**
 * quantize_q4_0_rows — W [N, K] (row-major, K % 32 == 0) into N·(K/32) blocks,
 * as ggml's quantize_row_q4_0_ref on each row. Returns false (out partly
 * written) if W holds a NaN or ±Inf, or a block's scale rounds to binary16
 * infinity (|max| / 8 >= 65520, i.e. |w| >= 524160).
 */
bool quantize_q4_0_rows(const float* W, int N, int K, std::uint8_t* out) noexcept;

/** Scalar table: exact int sums per block, FP32 accumulation. */
void gemv_q4_0_scalar(int N, int K, const float* x, const std::uint8_t* Wq, float* y);

// ─── SIMD tables (q4_0_avx2.cpp, q4_0_avxvnni.cpp, q4_0_avx512vnni.cpp) ──────
// The inner GEMVs have the Q8RowsFn signature and run under gemv_q8_0_run
// (q8_0.hpp), which quantizes x to Q8 blocks once per call.

/** AVX2 table: AVX-VNNI vpdpbusd if the CPU has it (and the unit was built), else vpmaddubsw. */
void gemv_q4_0_avx2(int N, int K, const float* x, const std::uint8_t* Wq, float* y);
/** AVX-512 table: AVX512_VNNI (EVEX vpdpbusd on YMM) if available, else gemv_q4_0_avx2. */
void gemv_q4_0_avx512(int N, int K, const float* x, const std::uint8_t* Wq, float* y);

/** The inner GEMVs, for tests (call only where the CPU has the ISA). */
void q4_0_rows_avx2(int N, int nb, const std::int8_t* xq, const float* dx,
                    const std::uint8_t* Wq, float* y) noexcept;
void q4_0_rows_avxvnni(int N, int nb, const std::int8_t* xq, const float* dx,
                       const std::uint8_t* Wq, float* y) noexcept;
void q4_0_rows_avx512vnni(int N, int nb, const std::int8_t* xq, const float* dx,
                          const std::uint8_t* Wq, float* y) noexcept;

}  // namespace quant
}  // namespace simd_ml
