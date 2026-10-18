#pragma once

/**
 * f16_gemm.hpp — FP16-input GEMM: C[fp32] = alpha · A[fp16] · B[fp16] + beta · C[fp32]
 *
 * A and B hold IEEE binary16 bit patterns (numpy.float16 memory), row-major;
 * accumulation and C are FP32. FP16 → FP32 is exact (F16C vcvtph2ps), so the
 * result is the FP32 GEMM of the converted inputs, with the same error bound.
 *
 * Implementation (avx2_gemm_packed.cpp, avx512_gemm_packed.cpp): the FP32
 * paths, with the conversion done once per element while packing A and B
 * (or, on the direct path for small products, once per 6-row strip of A and
 * in registers for B). The inner loop is the FP32 micro-kernel. docs/DESIGN.md §4.
 */

#include <cstdint>

namespace simd_ml {
namespace gemm {

/**
 * f16c_available — whether the CPU has F16C (and OS YMM state). Every AVX2
 * table requires it; the scalar table converts in software.
 */
bool f16c_available() noexcept;

/**
 * sgemm_f16_avx2 / sgemm_f16_avx512 — the per-ISA implementations; call
 * through kernels().sgemm_f16. M×K A (lda >= K), K×N B (ldb >= N), M×N C
 * (ldc >= N), in elements. beta = 0 overwrites C without reading it.
 * May throw std::bad_alloc (per-thread packing buffers, grown on first use).
 */
void sgemm_f16_avx2(int M, int N, int K, float alpha,
                    const std::uint16_t* A, int lda,
                    const std::uint16_t* B, int ldb,
                    float beta, float* C, int ldc);
void sgemm_f16_avx512(int M, int N, int K, float alpha,
                      const std::uint16_t* A, int lda,
                      const std::uint16_t* B, int ldb,
                      float beta, float* C, int ldc);

}  // namespace gemm
}  // namespace simd_ml
