/**
 * test_gemm.cpp — Numerical correctness tests for the AVX2/AVX-512 GEMM kernel
 * (avx_matmul.cpp).
 *
 * Tests the `simd_sgemm` function (6×16 AVX2 / 6×32 AVX-512 micro-kernel with
 * panel packing) against the triple-loop `scalar_sgemm` reference.
 *
 * ─── Error metric: combined absolute + relative bound ────────────────────────
 * Uses |got - ref| <= atol + rtol*|ref| (the same criterion np.allclose uses),
 * NOT naive relative error |got-ref|/(|ref|+tiny_floor). The naive form is a
 * real, previously-hit failure mode: random GEMM output can land arbitrarily
 * close to zero, and dividing a perfectly healthy absolute error by a near-zero
 * reference explodes into a spurious "large" relative error. This exact anti-
 * pattern caused a flaky CI failure in tests/test_precision.py's GeLU test
 * (see git history / CI log: max_rel_err=1.46e-4 against a 1e-4 threshold,
 * traced to an absolute error of 1.3e-7 divided by a reference of 3.4e-4).
 * GEMM outputs are equally vulnerable — this file uses the same fix.
 *
 * ─── Tolerance rationale ──────────────────────────────────────────────────────
 * FP32 accumulation over K multiplications has a rounding error budget of
 * O(K × ε_mach) where ε_mach ≈ 1.19e-7. We compile with -O3 -ffast-math,
 * which permits FP reassociation and can amplify this by O(log₂ K).
 * rtol scales with K (see tol_for_k below); atol is a fixed small floor that
 * absorbs near-zero-crossing noise without masking real bugs (a broken kernel
 * produces O(1)-O(100) absolute errors, dwarfing this floor by 5+ orders of
 * magnitude).
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "../src/kernels/cache_alloc.hpp"
#include "../src/kernels/avx_matmul.hpp"
#include "../src/dispatch/kernels.hpp"

static void fill_random(float* buf, int n, unsigned seed = 42) {
    // Linear congruential generator — reproducible, no stdlib dependency
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = static_cast<float>(static_cast<int>(s >> 16) % 200 - 100) * 0.01f;
    }
}

static double tol_for_k(int K) {
    return std::min(5e-2, static_cast<double>(K) * 1e-4);
}

// Combined absolute+relative bound check across an entire buffer.
// Returns the worst (diff - bound) value; <= 0 means every element passed.
static double max_bound_violation(const float* got, const float* ref, int n,
                                   double rtol, double atol,
                                   double* out_max_abs_err = nullptr) {
    double worst_violation = -1e300;
    double max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        double g = static_cast<double>(got[i]);
        double r = static_cast<double>(ref[i]);
        double diff  = std::fabs(g - r);
        double bound = atol + rtol * std::fabs(r);
        worst_violation = std::max(worst_violation, diff - bound);
        max_abs = std::max(max_abs, diff);
    }
    if (out_max_abs_err) *out_max_abs_err = max_abs;
    return worst_violation;
}

static bool test_gemm_size(int M, int N, int K) {
    const double rtol = tol_for_k(K);
    const double atol = 1e-4;   // absorbs near-zero-crossing noise; see file header

    auto A     = make_aligned_array<float>(M * K);
    auto B     = make_aligned_array<float>(K * N);
    auto C_ref = make_aligned_array<float>(M * N);
    auto C_got = make_aligned_array<float>(M * N);

    fill_random(A.get(), M * K, /*seed=*/1);
    fill_random(B.get(), K * N, /*seed=*/2);

    scalar_sgemm(M, N, K, A.get(), K, B.get(), N, C_ref.get(), N);
    simd_sgemm(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C_got.get(), N);

    double max_abs_err = 0.0;
    double violation = max_bound_violation(C_got.get(), C_ref.get(), M * N,
                                            rtol, atol, &max_abs_err);
    bool pass = violation <= 0.0;
    printf("  GEMM [%4d×%4d×%4d]  rtol=%.0e atol=%.0e  max_abs=%.2e  violation=%.2e  %s\n",
           M, N, K, rtol, atol, max_abs_err, violation, pass ? "PASS" : "FAIL");
    return pass;
}

// Test alpha/beta scaling paths
static bool test_gemm_alpha_beta() {
    const int M = 32, N = 32, K = 32;
    auto A = make_aligned_array<float>(M * K);
    auto B = make_aligned_array<float>(K * N);
    auto C = make_aligned_array<float>(M * N);
    auto C_ref = make_aligned_array<float>(M * N);

    fill_random(A.get(), M * K, 3);
    fill_random(B.get(), K * N, 4);
    fill_random(C.get(), M * N, 5);
    fill_random(C_ref.get(), M * N, 5);  // same initial C

    const float alpha = 2.5f, beta = 0.3f;

    auto AB = make_aligned_array<float>(M * N);
    scalar_sgemm(M, N, K, A.get(), K, B.get(), N, AB.get(), N);
    for (int i = 0; i < M * N; ++i)
        C_ref[i] = alpha * AB[i] + beta * C_ref[i];

    simd_sgemm(M, N, K, alpha, A.get(), K, B.get(), N, beta, C.get(), N);

    double max_abs_err = 0.0;
    double violation = max_bound_violation(C.get(), C_ref.get(), M * N,
                                            /*rtol=*/1e-3, /*atol=*/1e-4, &max_abs_err);
    bool pass = violation <= 0.0;
    printf("  alpha=%.1f beta=%.1f  [%dx%dx%d]  max_abs=%.2e  violation=%.2e  %s\n",
           alpha, beta, M, N, K, max_abs_err, violation, pass ? "PASS" : "FAIL");
    return pass;
}

/**
 * test_gemm_avx512_full_coverage — Forces 100% of the output through the
 * full-block micro-kernel path, with ZERO edge/tail blocks possible.
 *
 * M is an exact multiple of MR(6); N is an exact multiple of 32 (the
 * AVX-512 tile width) AND of 16 (the AVX2 tile width), so regardless of
 * which ISA this machine's simd_sgemm dispatches to, every single output
 * element is produced by the wide micro-kernel — none can be silently
 * "saved" by the scalar edge-block fallback masking a broken full-block path.
 *
 * This is the regression guard for the exact bug class that disabled
 * AVX-512 dispatch for an entire release cycle: the original
 * gemm_micro_6x32_avx512 stub only wrote the low 16 of every 32 output
 * columns, leaving the high 16 untouched. Untouched columns after a
 * beta=0 memset are exactly zero — which this test's random (non-zero)
 * reference data would immediately expose as a large diff in half the
 * matrix, impossible to miss.
 */
static bool test_gemm_avx512_full_coverage() {
    const int M = 60;    // = 10 * MR(6):  zero remainder rows
    const int N = 320;   // = 10 * 32 = 20 * 16: zero remainder cols either ISA
    const int K = 137;   // arbitrary, not a multiple of anything convenient

    auto A     = make_aligned_array<float>(M * K);
    auto B     = make_aligned_array<float>(K * N);
    auto C_ref = make_aligned_array<float>(M * N);
    auto C_got = make_aligned_array<float>(M * N);

    fill_random(A.get(), M * K, /*seed=*/11);
    fill_random(B.get(), K * N, /*seed=*/22);

    scalar_sgemm(M, N, K, A.get(), K, B.get(), N, C_ref.get(), N);
    simd_sgemm(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C_got.get(), N);

    const double rtol = tol_for_k(K);
    const double atol = 1e-4;
    double max_abs_err = 0.0;
    double violation = max_bound_violation(C_got.get(), C_ref.get(), M * N,
                                            rtol, atol, &max_abs_err);

    // Additionally verify the two halves of every 32-wide tile independently —
    // this is the exact split the original bug silently failed on (columns
    // 16-31 of each tile were never written).
    double max_abs_lo = 0.0, max_abs_hi = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int jtile = 0; jtile < N; jtile += 32) {
            for (int n = jtile; n < jtile + 16 && n < N; ++n)
                max_abs_lo = std::max(max_abs_lo,
                    (double)std::fabs(C_got[m*N+n] - C_ref[m*N+n]));
            for (int n = jtile + 16; n < jtile + 32 && n < N; ++n)
                max_abs_hi = std::max(max_abs_hi,
                    (double)std::fabs(C_got[m*N+n] - C_ref[m*N+n]));
        }
    }

    bool pass = violation <= 0.0;
    printf("  AVX-512 full-coverage [%dx%dx%d]  isa=%s  max_abs=%.2e  violation=%.2e\n",
           M, N, K, avx_matmul_isa_is_avx512() ? "avx512" : "avx2/scalar", max_abs_err, violation);
    printf("    lo-half (cols 0-15 of each 32-tile): max_abs=%.2e\n", max_abs_lo);
    printf("    hi-half (cols 16-31 of each 32-tile): max_abs=%.2e  %s\n",
           max_abs_hi, pass ? "PASS" : "FAIL");
    return pass;
}

int run_gemm_tests() {
    // simd_sgemm lives in an AVX2 unit (avx_matmul.cpp) and is called directly.
    if (!simd_ml::kernels_for(simd_ml::Isa::avx2)) {
        printf("\n── avx_matmul (simd_sgemm) tests: SKIP, CPU has no AVX2 ──\n");
        return 0;
    }
    printf("\n── GEMM Correctness Tests (avx_matmul.cpp) ──\n");
    bool all_pass = true;

    printf("  Runtime ISA: %s\n", avx_matmul_isa_is_avx512() ? "AVX-512 (6x32 kernel)" : "AVX2 (6x16 kernel)");

    // Correctness across sizes including edge and non-power-of-two
    all_pass &= test_gemm_size(1,   1,   1);
    all_pass &= test_gemm_size(8,   8,   8);
    all_pass &= test_gemm_size(16,  16,  16);
    all_pass &= test_gemm_size(64,  64,  64);
    all_pass &= test_gemm_size(128, 128, 128);
    all_pass &= test_gemm_size(256, 256, 256);
    all_pass &= test_gemm_size(100, 73,  89);   // non-power-of-2
    all_pass &= test_gemm_size(33,  17,  5);    // tiny irregular

    printf("  Alpha/Beta scaling:\n");
    all_pass &= test_gemm_alpha_beta();

    printf("  AVX-512 dual-accumulator regression guard:\n");
    all_pass &= test_gemm_avx512_full_coverage();

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
