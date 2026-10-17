/**
 * test_gemm_bf16.cpp — Correctness tests for sgemm_bf16_avx2 (BF16 inputs → FP32).
 *
 * Reference methodology (mirrors test_gemm_f16.cpp):
 *   1. Generate FP32 test data in [-0.5, +0.5].
 *   2. Convert to BF16 (right-shift 16, discard lower mantissa bits).
 *   3. Convert back to FP32 (zero-extend + left-shift 16) for the reference.
 *      Using the round-tripped values ensures quantisation noise is identical
 *      on both sides of the comparison — only FP32 accumulation order differs.
 *   4. Run naive_sgemm on those FP32 reference values → C_ref.
 *   5. Run sgemm_bf16_avx2 with the uint16_t BF16 data    → C_got.
 *   6. Compare: tolerance 1e-4 (BF16 ε ≈ 2⁻⁷ × 2 = ~1.56e-2 relative;
 *      since both sides use the same round-tripped inputs, the only delta
 *      is FP32 accumulation rounding — much smaller than the raw BF16 error).
 *
 * Additional test: bf16_avx512bf16_available() returns a bool without crashing
 * (runtime CPUID check; outcome depends on CPU, not a build-time assertion).
 *
 * No special compile flags are needed here: the BF16→FP32 helpers use only
 * AVX2 (_mm256_cvtepu16_epi32, _mm256_slli_epi32), which is always available
 * when TEST_SRCS_X86 is compiled (SIMD_ML_ENABLE_X86=ON).
 */

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <limits>

#include "../src/kernels/gemm/bf16_gemm.hpp"
#include "../src/kernels/gemm/naive_gemm.hpp"
#include "../src/kernels/cache_alloc.hpp"
#include "kernel_tables.hpp"

// The table the test functions below call into (set by the runner).
static const simd_ml::KernelTable* T = nullptr;

#ifdef __AVX2__
#  include <immintrin.h>
#endif

// ─── BF16 ↔ FP32 helpers (AVX2 integer — no F16C needed) ───────────────────

/** Convert FP32 to BF16: truncate lower 16 mantissa bits (round-to-zero). */
static uint16_t f32_to_bf16(float f) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

/** Convert BF16 to FP32: zero-extend and shift left 16 bits. */
static float bf16_to_f32(uint16_t b) noexcept {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// ─── Random fill ─────────────────────────────────────────────────────────────

static void fill_random_f32(float* buf, int n, unsigned seed) {
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = static_cast<float>(static_cast<int>(s >> 16) % 100 - 50) * 0.01f;
    }
}

// ─── Single test case ─────────────────────────────────────────────────────────

static bool test_bf16(int M, int N, int K,
                      float alpha = 1.0f, float beta = 0.0f,
                      unsigned seed_c = 0) {
    // Allocate FP32 source, convert to BF16
    auto A_f32 = make_aligned_array<float>(static_cast<std::size_t>(M) * K);
    auto B_f32 = make_aligned_array<float>(static_cast<std::size_t>(K) * N);
    fill_random_f32(A_f32.get(), M * K, 1);
    fill_random_f32(B_f32.get(), K * N, 2);

    auto A_bf16 = make_aligned_array<uint16_t>(static_cast<std::size_t>(M) * K);
    auto B_bf16 = make_aligned_array<uint16_t>(static_cast<std::size_t>(K) * N);
    for (int i = 0; i < M * K; ++i) A_bf16[i] = f32_to_bf16(A_f32[i]);
    for (int i = 0; i < K * N; ++i) B_bf16[i] = f32_to_bf16(B_f32[i]);

    // Reference: BF16 round-tripped back to FP32
    auto A_ref = make_aligned_array<float>(static_cast<std::size_t>(M) * K);
    auto B_ref = make_aligned_array<float>(static_cast<std::size_t>(K) * N);
    for (int i = 0; i < M * K; ++i) A_ref[i] = bf16_to_f32(A_bf16[i]);
    for (int i = 0; i < K * N; ++i) B_ref[i] = bf16_to_f32(B_bf16[i]);

    auto C_ref = make_aligned_array<float>(static_cast<std::size_t>(M) * N);
    auto C_got = make_aligned_array<float>(static_cast<std::size_t>(M) * N);
    if (seed_c != 0) {
        fill_random_f32(C_ref.get(), M * N, seed_c);
        std::copy(C_ref.get(), C_ref.get() + M * N, C_got.get());
    } else {
        std::fill(C_ref.get(), C_ref.get() + M * N, 0.0f);
        std::fill(C_got.get(), C_got.get() + M * N, 0.0f);
    }

    // Ground truth: FP32 GEMM on BF16-rounded inputs
    naive_sgemm(M, N, K,
                alpha, A_ref.get(), K, B_ref.get(), N,
                beta,  C_ref.get(), N);

    // Kernel under test
    T->sgemm_bf16(M, N, K,
                                    alpha, A_bf16.get(), K, B_bf16.get(), N,
                                    beta,  C_got.get(), N);

    // Tolerance 1e-4: both sides use the same BF16-rounded inputs so BF16
    // quantisation noise cancels. Remaining delta is FP32 accumulation order.
    float max_err = 0.0f;
    int   worst   = -1;
    for (int i = 0; i < M * N; ++i) {
        float e = std::fabs(C_got[i] - C_ref[i]);
        if (e > max_err) { max_err = e; worst = i; }
    }
    bool pass = (max_err < 1e-4f);

    if (alpha == 1.0f && beta == 0.0f) {
        printf("  bf16 [%4d×%4d×%4d]  max_abs=%.2e  %s\n",
               M, N, K, max_err, pass ? "PASS" : "FAIL");
    } else {
        printf("  bf16 [%4d×%4d×%4d] α=%.1f β=%.1f  max_abs=%.2e  %s\n",
               M, N, K, alpha, beta, max_err, pass ? "PASS" : "FAIL");
    }
    if (!pass && worst >= 0) {
        printf("    worst at [%d,%d]: got=%.6f  ref=%.6f\n",
               worst / N, worst % N, C_got[worst], C_ref[worst]);
    }
    return pass;
}

// ─── Test suite entry point ───────────────────────────────────────────────────

int run_bf16_gemm_tests() {
    printf("\n── BF16 GEMM Tests (sgemm_bf16_avx2, AVX2 zero-extend trick) ──\n");

    // Runtime check: bf16_avx512bf16_available() must not crash.
    // We don't require true/false — just that it returns without SIGILL.
    bool vdp = simd_ml::gemm::bf16_avx512bf16_available();
    printf("  bf16_avx512bf16_available(): %s (vdpbf16ps packing path: future v1.0)\n",
           vdp ? "true" : "false");

    const bool all_pass = for_each_kernel_table([](const simd_ml::KernelTable& t) {
        T = &t;
        bool ok = true;

        printf(" Square sizes (α=1, β=0):\n");
        for (int s : {1, 7, 8, 9, 16, 32, 64, 128}) {
            ok &= test_bf16(s, s, s);
        }

        printf(" Non-square / tail cases:\n");
        ok &= test_bf16(1,  64, 128);   // batch=1 inference (single row)
        ok &= test_bf16(4,  64, 256);   // small batch
        ok &= test_bf16(13, 31,  37);   // all tails in NR=8 loop
        ok &= test_bf16(32,  7,  17);   // wide M, narrow N

        printf(" Alpha/beta scaling:\n");
        ok &= test_bf16(32, 32, 32, 2.5f, 0.3f, /*seed_c=*/42);
        ok &= test_bf16(64, 64, 64, 0.5f, 1.0f, /*seed_c=*/55);

        printf(" beta=0 must zero C (not propagate NaN/inf):\n");
        {
            // Fill C with +infinity; after beta=0 call all elements must be finite.
            int Msz = 8, Nsz = 8, Ksz = 8;
            auto A = make_aligned_array<uint16_t>(Msz * Ksz);
            auto B = make_aligned_array<uint16_t>(Ksz * Nsz);
            auto C = make_aligned_array<float>(Msz * Nsz);
            for (int i = 0; i < Msz * Ksz; ++i) A[i] = f32_to_bf16(0.1f);
            for (int i = 0; i < Ksz * Nsz; ++i) B[i] = f32_to_bf16(0.1f);
            for (int i = 0; i < Msz * Nsz; ++i)
                C[i] = std::numeric_limits<float>::infinity();

            T->sgemm_bf16(Msz, Nsz, Ksz,
                                            1.0f, A.get(), Ksz, B.get(), Nsz,
                                            0.0f, C.get(), Nsz);
            bool finite_ok = true;
            for (int i = 0; i < Msz * Nsz; ++i)
                if (!std::isfinite(C[i])) { finite_ok = false; break; }
            printf("  bf16 beta=0 overwrites inf:  %s\n", finite_ok ? "PASS" : "FAIL");
            ok &= finite_ok;
        }

        printf(" BF16→FP32 bit-pattern correctness (unit test for conversion helper):\n");
        {
            // Verify the conversion by checking known BF16 values:
            //   BF16 = 0x3F80 → FP32 0x3F800000 = 1.0f
            //   BF16 = 0x4000 → FP32 0x40000000 = 2.0f
            //   BF16 = 0x0000 → FP32 0x00000000 = 0.0f
            //   BF16 = 0x7F80 → FP32 0x7F800000 = +inf
            struct { uint16_t bf16; float expected; } cases[] = {
                {0x3F80, 1.0f},
                {0x4000, 2.0f},
                {0x0000, 0.0f},
                {0xBF80, -1.0f},  // sign bit
            };
            bool conv_ok = true;
            for (auto& c : cases) {
                float got = bf16_to_f32(c.bf16);
                if (got != c.expected) {
                    printf("    bf16=0x%04X: expected %.1f got %.1f  FAIL\n",
                           c.bf16, c.expected, got);
                    conv_ok = false;
                }
            }
            printf("  BF16 bit-pattern unit tests:  %s\n", conv_ok ? "PASS" : "FAIL");
            ok &= conv_ok;
        }
        return ok;
    });

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
