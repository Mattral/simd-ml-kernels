/**
 * test_gemm_f16.cpp — Correctness tests for sgemm_f16_avx2 (FP16 inputs → FP32).
 *
 * Verifies sgemm_f16_avx2 against a reference FP32 GEMM built on the
 * FP32-converted values of the FP16 inputs — not the original FP32 source
 * data — so the only error source is FP32 accumulation rounding, not FP16
 * quantization noise. This makes the tolerance tight and meaningful.
 *
 * Test strategy:
 *   1. Generate FP32 test data in [-1, +1] with small magnitude (avoids
 *      FP16 overflow at ±65504, keeps test values representable).
 *   2. Round-trip through FP16 (fp32→fp16→fp32) to build the reference
 *      matrices; run naive_sgemm on those. This is the ground truth.
 *   3. Run sgemm_f16_avx2 with the uint16_t FP16 data.
 *   4. Compare outputs with tolerance 5e-4 (larger than FP32-only because
 *      repeated FP16→FP32 conversions accumulate 1 ULP each; at K=128
 *      worst-case ≈ 128 × 2⁻¹⁰ × max_val² ≈ 1.25e-4 per element).
 *
 * Runs on every kernel table the CPU supports; the FP16 conversions used to
 * build the inputs are done in software.
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include "../src/kernels/gemm/f16_gemm.hpp"
#include "../src/kernels/gemm/naive_gemm.hpp"
#include "../src/kernels/cache_alloc.hpp"

#include "kernel_tables.hpp"

// The table the test functions below call into (set by the runner).
static const simd_ml::KernelTable* T = nullptr;

// ─── FP16 <-> FP32 in software ───────────────────────────────────────────────
// No F16C here: the test binary is built for baseline x86-64 and must run on
// CPUs without it (the scalar table is tested there).

static float bits_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t f32_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

/** FP32 -> FP16 bit pattern, round to nearest even; finite inputs below 65504. */
static uint16_t f32_to_f16(float f) noexcept {
    const uint32_t u = f32_bits(f);
    const uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000u);
    const float a = std::fabs(f);
    if (a < 6.103515625e-05f) {                   // FP16 subnormal (< 2^-14) or zero
        // In units of the FP16 subnormal step 2^-24; nearbyint rounds half to even.
        return static_cast<uint16_t>(sign | static_cast<uint16_t>(std::nearbyint(a * 16777216.0f)));
    }
    uint32_t man = u & 0x7fffffu, exp = ((u >> 23) & 0xffu) - 127 + 15;
    uint32_t h = (exp << 10) | (man >> 13);
    const uint32_t rest = man & 0x1fffu;          // 13 dropped bits
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;   // carries into exp correctly
    return static_cast<uint16_t>(sign | h);
}

/** FP16 bit pattern -> FP32 (normal and subnormal inputs). */
static float f16_to_f32(uint16_t h) noexcept {
    const float sign = (h & 0x8000u) ? -1.0f : 1.0f;
    const int exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    if (exp == 0) return sign * std::ldexp(static_cast<float>(man), -24);
    return sign * bits_f32(static_cast<uint32_t>(exp - 15 + 127) << 23 | static_cast<uint32_t>(man) << 13);
}

// ─── Random fill ─────────────────────────────────────────────────────────────

static void fill_random_f32(float* buf, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        // Values in [-0.5, +0.5]: small enough to be exactly representable
        // in FP16 (range ±65504), large enough to stress FMA accumulation.
        buf[i] = (float)(int(s >> 16) % 100 - 50) * 0.01f;
    }
}

// ─── Single test case ─────────────────────────────────────────────────────────

static bool test_f16(int M, int N, int K,
                     float alpha = 1.0f, float beta = 0.0f,
                     unsigned seed_c = 0) {
    // Allocate FP32 source data
    auto A_f32  = make_aligned_array<float>(static_cast<std::size_t>(M) * K);
    auto B_f32  = make_aligned_array<float>(static_cast<std::size_t>(K) * N);
    fill_random_f32(A_f32.get(), M * K, 1);
    fill_random_f32(B_f32.get(), K * N, 2);

    // Allocate FP16 inputs (uint16_t bit patterns) by converting FP32 → FP16
    auto A_f16 = make_aligned_array<uint16_t>(static_cast<std::size_t>(M) * K);
    auto B_f16 = make_aligned_array<uint16_t>(static_cast<std::size_t>(K) * N);
    for (int i = 0; i < M * K; ++i) A_f16[i] = f32_to_f16(A_f32[i]);
    for (int i = 0; i < K * N; ++i) B_f16[i] = f32_to_f16(B_f32[i]);

    // Build FP32 reference matrices from the FP16 round-trip values.
    // This correctly accounts for FP16 quantisation noise in the reference.
    auto A_ref_f32 = make_aligned_array<float>(static_cast<std::size_t>(M) * K);
    auto B_ref_f32 = make_aligned_array<float>(static_cast<std::size_t>(K) * N);
    for (int i = 0; i < M * K; ++i) A_ref_f32[i] = f16_to_f32(A_f16[i]);
    for (int i = 0; i < K * N; ++i) B_ref_f32[i] = f16_to_f32(B_f16[i]);

    // Reference C (FP32) and kernel C (FP32)
    auto C_ref = make_aligned_array<float>(static_cast<std::size_t>(M) * N);
    auto C_got = make_aligned_array<float>(static_cast<std::size_t>(M) * N);
    if (seed_c != 0) {
        fill_random_f32(C_ref.get(), M * N, seed_c);
        std::copy(C_ref.get(), C_ref.get() + M * N, C_got.get());
    } else {
        std::fill(C_ref.get(), C_ref.get() + M * N, 0.0f);
        std::fill(C_got.get(), C_got.get() + M * N, 0.0f);
    }

    // Ground truth: FP32 GEMM on the FP16-rounded inputs
    naive_sgemm(M, N, K,
                alpha, A_ref_f32.get(), K, B_ref_f32.get(), N,
                beta,  C_ref.get(), N);

    // Kernel under test
    T->sgemm_f16(M, N, K,
                 alpha, A_f16.get(), K, B_f16.get(), N,
                 beta,  C_got.get(), N);

    // Error analysis
    // Tolerance 5e-4: FP16 epsilon ≈ 2⁻¹⁰ ≈ 9.8e-4; at K≤128 and
    // element magnitude ≤ 0.5, per-output worst case ≈ K × 2⁻¹⁰ × 0.5²
    // ≈ 128 × 9.8e-4 × 0.25 ≈ 3.1e-2. But both sides of the comparison
    // used the SAME FP32-converted inputs, so quantisation cancels and
    // only FP32 accumulation order differs → 5e-4 is generous.
    float max_err = 0.0f;
    int   worst   = -1;
    for (int i = 0; i < M * N; ++i) {
        float e = std::fabs(C_got[i] - C_ref[i]);
        if (e > max_err) { max_err = e; worst = i; }
    }

    bool pass = (max_err < 5e-4f);
    if (alpha == 1.0f && beta == 0.0f) {
        printf("  f16 [%4d×%4d×%4d]  max_abs=%.2e  %s\n",
               M, N, K, max_err, pass ? "PASS" : "FAIL");
    } else {
        printf("  f16 [%4d×%4d×%4d] α=%.1f β=%.1f  max_abs=%.2e  %s\n",
               M, N, K, alpha, beta, max_err, pass ? "PASS" : "FAIL");
    }
    if (!pass && worst >= 0) {
        printf("    worst at [%d,%d]: got=%.6f  ref=%.6f\n",
               worst / N, worst % N, C_got[worst], C_ref[worst]);
    }
    return pass;
}

// ─── Test suite entry point ───────────────────────────────────────────────────

int run_f16_gemm_tests() {
    printf("\n── FP16 GEMM Tests (KernelTable::sgemm_f16) ──\n");

    const bool all_pass = for_each_kernel_table([](const simd_ml::KernelTable& t) {
        T = &t;
        bool ok = true;

        printf(" Square sizes (α=1, β=0):\n");
        for (int s : {1, 7, 8, 9, 16, 32, 64, 128}) {
            ok &= test_f16(s, s, s);
        }

        printf(" Non-square / tail cases:\n");
        ok &= test_f16(1,  64, 128);   // single-row (batch=1 inference)
        ok &= test_f16(4,  64, 256);   // small batch
        ok &= test_f16(13, 31,  37);   // all tails
        ok &= test_f16(32,  7,  17);   // wide M, narrow N

        printf(" Alpha/beta scaling:\n");
        ok &= test_f16(32, 32, 32, 2.5f, 0.3f, /*seed_c=*/99);
        ok &= test_f16(64, 64, 64, 0.5f, 1.0f, /*seed_c=*/77);

        printf(" beta=0 must overwrite (not multiply) existing C:\n");
        {
            // Fill C with NaN; after beta=0 call all elements must be finite.
            auto A = make_aligned_array<uint16_t>(8 * 8);
            auto B = make_aligned_array<uint16_t>(8 * 8);
            auto C = make_aligned_array<float>(8 * 8);
            for (int i = 0; i < 64; ++i) {
                A[i] = f32_to_f16(0.1f);
                B[i] = f32_to_f16(0.1f);
                // Poison C with positive infinity
                C[i] = std::numeric_limits<float>::infinity();
            }
            T->sgemm_f16(8, 8, 8,
                         1.0f, A.get(), 8, B.get(), 8,
                         0.0f, C.get(), 8);
            bool finite_ok = true;
            for (int i = 0; i < 64; ++i)
                if (!std::isfinite(C[i])) { finite_ok = false; break; }
            printf("  f16 beta=0 overwrites inf:  %s\n", finite_ok ? "PASS" : "FAIL");
            ok &= finite_ok;
        }
        return ok;
    });

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
