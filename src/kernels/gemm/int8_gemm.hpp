#pragma once

/**
 * int8_gemm.hpp — INT8 GEMM with per-channel scales, FP32 output:
 *
 *   C[i][j] = a_scale · b_scales[j] · Σ_k (A[i][k] − a_zero_point) · op(B)[k][j]
 *
 * A is uint8 (activations, per-tensor scale and zero point), B is int8
 * (weights, symmetric, one scale per output column j), as in PyTorch's
 * quantized linear (quint8 × per-channel qint8). op(B) = B (stored K×N, ldb
 * >= N) or Bᵀ (trans_b: stored N×K, ldb >= K, as torch.nn.Linear keeps W).
 * C is overwritten (no alpha/beta).
 *
 * Paths (docs/DESIGN.md §4.2):
 *  - AVX-VNNI (vpdpbusd on YMM; Alder Lake, Lunar Lake, Sapphire Rapids):
 *    avxvnni_gemm.cpp, 6×16 int32 tile;
 *  - AVX512_VNNI (Cascade Lake and later, Zen 4 and later): avx512vnni_gemm.cpp,
 *    12×32 int32 tile;
 *  - otherwise (AVX2 without VNNI): A − zero point and B converted to FP32 while
 *    packing, the FP32 6×16 kernel, then the column scales; same speed class as
 *    the FP32 GEMM, a quarter of its operand memory;
 *  - scalar table: portable C++, exact int64 sums.
 * The VNNI paths sum exactly in int32 within one K block (|sum| < 2^31 for
 * blocks up to 65793 values) and accumulate the scaled blocks in FP32.
 * Error bound, every path: |C − C64| ≤ 2·(K+2)·u·|a_scale·b_scales[j]|·Σ_k |A − zp|·|B|,
 * u = 2^-24, against the exact (float64) value.
 */

#include <cstdint>

namespace simd_ml {
namespace gemm {

/** CPUID: AVX-VNNI / AVX512_VNNI, with the OS register-state checks. */
bool avx_vnni_available() noexcept;
bool avx512vnni_available() noexcept;

/**
 * u8s8_path — the path kernels().sgemm_u8s8 takes on this CPU and build:
 * "avx512_vnni", "avx_vnni", "fp32_convert" or "scalar".
 */
const char* u8s8_path() noexcept;

// Every implementation takes the same arguments (KernelTable::sgemm_u8s8,
// dispatch/kernels.hpp). May throw std::bad_alloc (per-thread packing buffers).

/** AVX2 table: AVX-VNNI if the CPU has it (and the unit was built), else the FP32 conversion path. */
void sgemm_u8s8_avx2(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                     int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                     const float* b_scales, float* C, int ldc);
/** The FP32 conversion path on its own (AVX2 unit), for tests and the benchmark. */
void sgemm_u8s8_convert_avx2(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                             int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                             const float* b_scales, float* C, int ldc);
/** AVX-VNNI path (avxvnni_gemm.cpp). Call only if avx_vnni_available(). */
void sgemm_u8s8_avxvnni(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                        int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                        const float* b_scales, float* C, int ldc);
/** AVX-512 table: AVX512_VNNI if available, else sgemm_u8s8_avx2. */
void sgemm_u8s8_avx512(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                       int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                       const float* b_scales, float* C, int ldc);
/** AVX512_VNNI path (avx512vnni_gemm.cpp). Call only if avx512vnni_available(). */
void sgemm_u8s8_avx512vnni(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                           int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                           const float* b_scales, float* C, int ldc);

}  // namespace gemm
}  // namespace simd_ml
