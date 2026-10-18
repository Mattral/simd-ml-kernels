#pragma once

/**
 * bf16_gemm.hpp — BF16-input GEMM: C[fp32] = alpha · A[bf16] · B[bf16] + beta · C[fp32]
 *
 * BF16 is the top 16 bits of an FP32 (same 8-bit exponent, 7 mantissa bits),
 * so BF16 → FP32 is exact: zero-extend to 32 bits and shift left 16. The
 * result is the FP32 GEMM of the converted inputs, with the same error bound.
 *
 * Implementation: as sgemm_f16 (f16_gemm.hpp): the FP32 paths with the
 * conversion done while packing, the FP32 micro-kernel in the inner loop.
 * docs/DESIGN.md §4.
 */

#include <cstdint>

namespace simd_ml {
namespace gemm {

/**
 * bf16_avx512bf16_available — whether the CPU has AVX512_BF16 (vdpbf16ps;
 * CPUID leaf 7 subleaf 1 EAX bit 5) and OS ZMM state.
 */
bool bf16_avx512bf16_available() noexcept;

/**
 * bf16_dot_enabled — whether sgemm_bf16 may take the vdpbf16ps path: AVX512_BF16
 * hardware and the opt-in SIMD_ML_BF16_DOT=1 (read once). Off by default: the
 * conversion path was faster where measured (docs/DESIGN.md §4.1).
 * set_bf16_dot — override the environment (tests).
 */
bool bf16_dot_enabled() noexcept;
void set_bf16_dot(bool on) noexcept;

/**
 * sgemm_bf16_avx2 / sgemm_bf16_avx512 — the per-ISA implementations; call
 * through kernels().sgemm_bf16. Same layout, beta and exception rules as
 * sgemm_f16_avx2.
 */
void sgemm_bf16_avx2(int M, int N, int K, float alpha,
                     const std::uint16_t* A, int lda,
                     const std::uint16_t* B, int ldb,
                     float beta, float* C, int ldc);
void sgemm_bf16_avx512(int M, int N, int K, float alpha,
                       const std::uint16_t* A, int lda,
                       const std::uint16_t* B, int ldb,
                       float beta, float* C, int ldc);

/**
 * sgemm_bf16_avx512bf16 — the vdpbf16ps path (avx512bf16_gemm.cpp), compiled
 * when the compiler supports -mavx512bf16 (SIMD_ML_HAVE_AVX512BF16). Call only
 * if bf16_avx512bf16_available(); sgemm_bf16_avx512 checks bf16_dot_enabled(). Unlike
 * the conversion path, BF16 subnormal inputs are read as zero and subnormal
 * results are flushed to zero (the instruction ignores MXCSR).
 */
void sgemm_bf16_avx512bf16(int M, int N, int K, float alpha,
                           const std::uint16_t* A, int lda,
                           const std::uint16_t* B, int ldb,
                           float beta, float* C, int ldc);

}  // namespace gemm
}  // namespace simd_ml
