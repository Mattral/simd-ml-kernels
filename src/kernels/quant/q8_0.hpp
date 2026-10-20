#pragma once

/**
 * q8_0.hpp — ggml's Q8_0 weight format and the decode GEMV on it
 * (docs/DESIGN.md §2.8).
 *
 * Format (byte-identical to ggml's block_q8_0): a row of K values is K/32
 * blocks of 34 bytes, each an IEEE binary16 scale d (little-endian) followed
 * by 32 int8 quants q; value i of the block is d·q[i]. A matrix W [N, K] is
 * its N rows one after the other, N·(K/32)·34 bytes. Quantization is ggml's
 * quantize_row_q8_0_ref: per block amax = max |w|, d = amax / 127 (FP32),
 * q = roundf(w · (1/d)) (half away from zero; 0 if d = 0), and the stored
 * scale is d rounded to binary16 (round to nearest even). One case differs:
 * a block with amax below 127/FLT_MAX (1/d overflows) stores q = 0, where the
 * C reference converts ±inf or NaN to int8 (undefined; 0 on x86); its scale
 * is binary16 zero either way.
 *
 *   gemv_q8_0:  y[n] = Σ_b fp16(dw[n][b]) · dx[b] · Σ_i qw[n][b][i] · qx[b][i]
 *
 * x is quantized once per call with the same rule (its qs equal
 * quantize_q8_0(x)'s), but its scale dx stays FP32: no binary16 rounding,
 * underflow or overflow for activations. The 32-term integer sums are exact
 * (|Σ| <= 32·127·127). Error bound of every path, against that sum taken
 * exactly (float64):
 *
 *   |y − y64| <= 2·(K/32 + 4)·u·Σ_b |fp16(dw_b)·dx_b|·Σ_i |qw·qx|,  u = 2^-24.
 *
 * x with a NaN or ±Inf: every y is NaN. Scales in Wq that are not finite
 * (only possible in bytes not made by quantize_q8_0_rows): unspecified.
 */

#include <cstddef>
#include <cstdint>

namespace simd_ml {
namespace quant {

constexpr int kQ8_0Values = 32;   ///< values per block
constexpr int kQ8_0Bytes  = 34;   ///< bytes per block (binary16 scale + 32 int8)

/** Bytes of one Q8_0 row of K values (K % 32 == 0). */
constexpr std::size_t q8_0_row_bytes(int K) {
    return static_cast<std::size_t>(K / kQ8_0Values) * kQ8_0Bytes;
}

// ─── Scalar unit (q8_0_scalar.cpp; no -ffast-math, so IEEE rounding is exact) ─

/** IEEE binary32 -> binary16, round to nearest even (NaN -> 0x7e00 with the sign). */
std::uint16_t fp32_to_fp16(float f) noexcept;
/** IEEE binary16 -> binary32, exact. */
float fp16_to_fp32(std::uint16_t h) noexcept;

/**
 * quantize_q8_0_rows — W [N, K] (row-major, K % 32 == 0) into N·(K/32) blocks,
 * as ggml's quantize_row_q8_0_ref on each row. Returns false (out partly
 * written) if W holds a NaN or ±Inf, or a block's scale rounds to binary16
 * infinity (amax / 127 >= 65520, i.e. |w| >= about 8.32e6).
 */
bool quantize_q8_0_rows(const float* W, int N, int K, std::uint8_t* out) noexcept;

/**
 * quantize_x_q8 — x [K] into K int8 quants (as quantize_q8_0_rows) and K/32
 * FP32 scales dx = amax / 127. Returns false if x holds a NaN or ±Inf.
 */
bool quantize_x_q8(const float* x, int K, std::int8_t* qs, float* dx) noexcept;

/**
 * Q8RowsFn — the inner GEMV of one ISA: y[0..N) from x's nb blocks (qs, dx)
 * and the Q8_0 rows of Wq. No allocation, no checks.
 */
using Q8RowsFn = void (*)(int N, int nb, const std::int8_t* xq, const float* dx,
                          const std::uint8_t* Wq, float* y) noexcept;

/**
 * gemv_q8_0_run — the shared driver of gemv_q8_0 and gemv_q4_0: quantizes x
 * into per-thread scratch (may throw std::bad_alloc), then calls rows on W's
 * rows of block_bytes-byte blocks (34: Q8_0, 18: Q4_0). In OpenMP builds the
 * rows are split into one contiguous slice per thread (gemm::gemv_threads,
 * gemv_slice: whole 8-row groups); x is quantized by the team (one run of
 * blocks per thread, then a barrier) into the same quants one thread writes,
 * and each output is computed by one thread exactly as with one thread. K % 32 == 0 is
 * the caller's check; K == 0 gives y = 0.
 */
void gemv_q8_0_run(int N, int K, const float* x, const std::uint8_t* Wq, float* y, Q8RowsFn rows,
                   int block_bytes);

/** Scalar table: exact int sums per block, FP32 accumulation. */
void gemv_q8_0_scalar(int N, int K, const float* x, const std::uint8_t* Wq, float* y);

// ─── SIMD tables (q8_0_avx2.cpp, q8_0_avxvnni.cpp, q8_0_avx512vnni.cpp) ──────

/** AVX2 table: AVX-VNNI vpdpbusd if the CPU has it (and the unit was built), else vpmaddubsw. */
void gemv_q8_0_avx2(int N, int K, const float* x, const std::uint8_t* Wq, float* y);
/** AVX-512 table: AVX512_VNNI (EVEX vpdpbusd on YMM) if available, else gemv_q8_0_avx2. */
void gemv_q8_0_avx512(int N, int K, const float* x, const std::uint8_t* Wq, float* y);

/** The inner GEMVs, for tests (call only where the CPU has the ISA). */
void q8_0_rows_avx2(int N, int nb, const std::int8_t* xq, const float* dx,
                    const std::uint8_t* Wq, float* y) noexcept;
void q8_0_rows_avxvnni(int N, int nb, const std::int8_t* xq, const float* dx,
                       const std::uint8_t* Wq, float* y) noexcept;
void q8_0_rows_avx512vnni(int N, int nb, const std::int8_t* xq, const float* dx,
                          const std::uint8_t* Wq, float* y) noexcept;

/**
 * q8_0_path — the inner GEMV kernels().gemv_q8_0 uses on this CPU and build:
 * "avx512_vnni", "avx_vnni", "avx2" or "scalar".
 */
const char* q8_0_path() noexcept;

}  // namespace quant
}  // namespace simd_ml
