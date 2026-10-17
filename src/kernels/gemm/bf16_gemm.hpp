#pragma once

/**
 * bf16_gemm.hpp — BF16-input GEMM with FP32 accumulation (AVX2 + optional AVX-512 BF16)
 *
 * Computes: C[fp32] = alpha * A[bf16] @ B[bf16] + beta * C[fp32]
 *
 * ─── Why BF16? ────────────────────────────────────────────────────────────────
 * BF16 (Brain Float 16) uses the same 8-bit exponent as FP32, giving the same
 * dynamic range (vs FP16's 5-bit exponent capped at ±65504). The trade-off is
 * lower mantissa precision (7 bits vs 10 for FP16, 23 for FP32). BF16 is the
 * dominant reduced-precision format for LLM training (gradient accumulation
 * stays stable) and is the native format for Google TPU, NVIDIA A100+ (via
 * HFMA2 / HMMA16816), and Intel AMX / Sapphire Rapids (TDPBF16PS).
 *
 * ─── BF16→FP32 conversion: zero-extend + shift (AVX2, no special ISA) ────────
 * BF16 is defined as the top 16 bits of FP32 (same exponent, truncated mantissa):
 *
 *   FP32 bit layout: [S:1][Exp:8][Mant:23]
 *   BF16 bit layout: [S:1][Exp:8][Mant: 7]  ← same S+Exp, 7 MSBs of Mant
 *
 * Converting BF16 → FP32 is therefore:
 *   1. Zero-extend uint16_t → uint32_t  (_mm256_cvtepu16_epi32)
 *   2. Left-shift 16 bits               (_mm256_slli_epi32, 16)
 *   3. Reinterpret as float             (_mm256_castsi256_ps)
 *
 * This is 2–3 AVX2 integer instructions per 8 values — cheaper than F16C's
 * VCVTPH2PS, and requires NO special extension beyond AVX2+FMA which is
 * already the project's minimum ISA.
 *
 * ─── vdpbf16ps (AVX-512 BF16) path ──────────────────────────────────────────
 * Intel Ice Lake-SP (Xeon, 2021+) / Sapphire Rapids / AMD Zen4+ support
 * VDPBF16PS which accumulates two BF16 pairs per cycle into FP32. It requires
 * -mavx512bf16 (CPUID EAX=7, ECX=1, EBX bit 5). Check bf16_avx512bf16_available().
 *
 * The vdpbf16ps path also requires panel packing to interleave B rows in
 * pairs — this is implemented in ROADMAP.md v1.0, not here. The current
 * implementation uses the AVX2 zero-extend path on all hardware.
 *
 * ─── See also ─────────────────────────────────────────────────────────────────
 * docs/ROADMAP.md §v0.9, docs/DESIGN.md §4, f16_gemm.hpp (FP16 variant)
 */

#include <cstdint>

namespace simd_ml {
namespace gemm {

// ─── Runtime capability checks ───────────────────────────────────────────────

/**
 * bf16_avx512bf16_available — Returns true if CPU supports AVX-512 BF16
 * (vdpbf16ps / TDPBF16PS instructions, CPUID leaf 7 subleaf 1 EBX bit 5).
 *
 * Available on Intel Ice Lake-SP (2021+), Cooper Lake, Sapphire Rapids,
 * and AMD Zen4+ (EPYC Genoa). Not required for sgemm_bf16_avx2 — checked
 * here for informational / future dispatch purposes.
 */
bool bf16_avx512bf16_available() noexcept;

// ─── Core kernel ──────────────────────────────────────────────────────────────

/**
 * sgemm_bf16_avx2 — BF16-input GEMM with FP32 accumulation (AVX2 only).
 *
 * Computes: C[float32] = alpha * A[bf16] @ B[bf16] + beta * C[float32]
 *
 * A and B are stored as uint16_t arrays in BF16 format. C is always FP32.
 * Conversion from BF16→FP32 uses only AVX2 integer instructions
 * (_mm256_cvtepu16_epi32 + _mm256_slli_epi32) — no F16C or AVX-512 required.
 *
 * Implements the same direct (no-packing) row-by-row AVX2 FMA structure as
 * sgemm_direct_avx2 and sgemm_f16_avx2. Suitable for batch sizes 1–128 where
 * A+B+C fits in L2 cache and hardware prefetch handles the strided B access.
 *
 * Parameters
 * ----------
 * M, N, K — matrix dimensions: A is [M×K], B is [K×N], C is [M×N]
 * alpha   — scalar multiplier for A@B product
 * A       — BF16 matrix, row-major, leading dim lda (in uint16_t units)
 * lda     — elements between A rows (normally = K)
 * B       — BF16 matrix, row-major, leading dim ldb (in uint16_t units)
 * ldb     — elements between B rows (normally = N)
 * beta    — scalar multiplier for existing C (0.0 = overwrite)
 * C       — FP32 output, row-major, leading dim ldc (in float units)
 * ldc     — elements between C rows (normally = N)
 */
void sgemm_bf16_avx2(int M, int N, int K,
                     float alpha,
                     const uint16_t* __restrict__ A, int lda,
                     const uint16_t* __restrict__ B, int ldb,
                     float beta,
                     float*          __restrict__ C, int ldc) noexcept;

}  // namespace gemm
}  // namespace simd_ml
