#pragma once

/**
 * f16_gemm.hpp — FP16-input GEMM with FP32 accumulation (F16C + AVX2)
 *
 * Computes: C[fp32] = alpha * A[fp16] @ B[fp16] + beta * C[fp32]
 *
 * ─── Why FP16 inputs + FP32 accumulation? ────────────────────────────────────
 * Modern transformer inference stores weights and activations in FP16 to halve
 * memory bandwidth. However, accumulating FP16 dot products into an FP16
 * accumulator loses 1–2 ULP per step, causing visible divergence at K > 64.
 * FP32 accumulation gives full precision at negligible runtime cost (the
 * conversion is 1 instruction per 8 values, hidden behind the FMA latency).
 *
 * ─── Instruction set: F16C + AVX2 ────────────────────────────────────────────
 * F16C (available on all Haswell+ and AMD Piledriver+ CPUs — i.e. every CPU
 * that also has AVX2) provides:
 *   _mm256_cvtph_ps(__m128i) — convert 8 packed FP16 → 8 FP32 (__m256)
 *   _mm256_cvtps_ph(__m256, rounding) — convert 8 FP32 → 8 packed FP16
 *   _cvtsh_ss(uint16_t)       — convert 1 FP16 → FP32 scalar
 *
 * A and B are stored as raw uint16_t arrays (IEEE 754 half-precision bit
 * patterns — the same layout used by numpy.float16).
 *
 * ─── See also ─────────────────────────────────────────────────────────────────
 * docs/ROADMAP.md §v0.9, docs/DESIGN.md §4
 */

#include <cstdint>

namespace simd_ml {
namespace gemm {

// ─── Runtime capability check ─────────────────────────────────────────────────
/**
 * f16c_available — Returns true if the CPU supports F16C instructions.
 *
 * All CPUs that have AVX2 also have F16C in practice (Intel Haswell/Broadwell,
 * AMD Piledriver+). However, we check explicitly so callers can raise a clean
 * error rather than SIGILL if somehow running on a CPU that lacks it.
 *
 * Uses CPUID + XGETBV at runtime via dispatch::CpuFeatures (compile-time
 * -mf16c enables the intrinsics; the runtime check guards the dispatch).
 */
bool f16c_available() noexcept;

// ─── Core kernel ──────────────────────────────────────────────────────────────
/**
 * sgemm_f16_avx2 — Half-precision input GEMM, FP32 output / accumulation.
 *
 * Computes: C[float32] = alpha * A[fp16] @ B[fp16] + beta * C[float32]
 *
 * A and B are stored as uint16_t arrays in IEEE 754 FP16 format (= numpy
 * float16 memory layout). C is FP32 throughout — no conversion on output.
 *
 * Implementation: row-by-row AVX2 FMA loop, F16C conversion on load (no
 * packing), matching the sgemm_direct_avx2 structure. Suitable for the
 * inference workloads this ISA pair targets (batch sizes 1–128 where
 * everything fits in L2 cache; packing is not beneficial at those scales).
 *
 * Requires F16C + AVX2 + FMA. Call f16c_available() before using.
 *
 * Parameters
 * ----------
 * M, N, K  — matrix dimensions: A is [M×K], B is [K×N], C is [M×N]
 * alpha    — scalar multiplier for A@B product
 * A        — FP16 matrix, row-major, leading dimension lda (in uint16_t units)
 * lda      — distance in uint16_t elements between rows of A (normally = K)
 * B        — FP16 matrix, row-major, leading dimension ldb (in uint16_t units)
 * ldb      — distance in uint16_t elements between rows of B (normally = N)
 * beta     — scalar multiplier for existing C (0.0 = overwrite)
 * C        — FP32 matrix, row-major, leading dimension ldc (in float units)
 * ldc      — distance in float elements between rows of C   (normally = N)
 */
void sgemm_f16_avx2(int M, int N, int K,
                    float alpha,
                    const uint16_t* __restrict__ A, int lda,
                    const uint16_t* __restrict__ B, int ldb,
                    float beta,
                    float*          __restrict__ C, int ldc) noexcept;

}  // namespace gemm
}  // namespace simd_ml
