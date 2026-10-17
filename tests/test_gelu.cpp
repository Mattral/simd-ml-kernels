/**
 * test_gelu.cpp — GeLU Correctness Tests
 *
 * Two distinct correctness criteria:
 *
 * 1. AVX2 fast path (gelu_forward_avx2) vs tanh-formula reference:
 *    GeLU_tanh(x) = 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))
 *    Expected tolerance: < 2e-5  (Cody–Waite exp error budget × scaling)
 *
 * 2. AVX2 exact path (gelu_erf_forward_avx2) vs float64 erf reference:
 *    GeLU_erf(x) = 0.5*x*(1 + erf(x/√2))
 *    Bound: 2e-6 absolute on [-10, 10] (the bound in the Python docstring).
 *
 * 3. Scalar exact path (gelu_forward_scalar) vs erff reference:
 *    Expected tolerance: < 1e-5  (FP32 rounding only)
 *
 * 4. Both AVX2 paths: special values (±inf, NaN, ±0, subnormals) in the 8-wide
 *    body and in the masked tail, and tail results bitwise equal to the body.
 *
 * Note: gelu_forward_avx2 (tanh formula) and gelu_forward_scalar (erf formula)
 * are NOT expected to match each other to < 1e-4 — they implement different
 * mathematical approximations of the same function. The max abs difference
 * between the two formulas is ~5e-4 at x ≈ ±2.7, regardless of implementation
 * accuracy. This is documented in DESIGN.md and the preprint (Section 6.4).
 */

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include "../src/kernels/cache_alloc.hpp"
#include "../src/kernels/intrinsic_gelu.hpp"

// Reference implementation of the TANH-approximation GeLU (not erff)
// Used to test the AVX2 fast path to correct tolerances.
static float gelu_tanh_ref(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float COEFF = 0.044715f;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

// Reference: exact erf GeLU
static float gelu_erf_ref(float x) {
    constexpr float INV_SQRT2 = 0.7071067811865476f;
    return 0.5f * x * (1.0f + std::erff(x * INV_SQRT2));
}

static bool test_gelu_avx2_vs_tanh_ref(int n, float lo, float hi,
                                         double tol = 2e-5) {
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    for (int i = 0; i < n; ++i)
        inp[i] = lo + (hi - lo) * i / std::max(n - 1, 1);

    gelu_forward_avx2(inp.get(), out.get(), n);

    double max_abs = 0.0;
    int worst_idx = 0;
    for (int i = 0; i < n; ++i) {
        double ref = gelu_tanh_ref(inp[i]);
        double err = std::abs((double)out[i] - ref);
        if (err > max_abs) { max_abs = err; worst_idx = i; }
    }
    bool pass = (max_abs < tol);
    printf("  avx2_vs_tanh_ref  [%5.1f,%5.1f] n=%-7d  max_abs_err=%.2e  %s",
           lo, hi, n, max_abs, pass ? "PASS" : "FAIL");
    if (!pass)
        printf("  (worst x=%.4f out=%.6f ref=%.6f)",
               (double)inp[worst_idx], (double)out[worst_idx],
               (double)gelu_tanh_ref(inp[worst_idx]));
    printf("\n");
    return pass;
}

static bool test_gelu_erf_avx2_vs_ref(int n, float lo, float hi, double tol = 2e-6) {
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    for (int i = 0; i < n; ++i)
        inp[i] = lo + (hi - lo) * i / std::max(n - 1, 1);

    gelu_erf_forward_avx2(inp.get(), out.get(), n);

    double max_abs = 0.0;
    int worst = 0;
    for (int i = 0; i < n; ++i) {
        const double x = inp[i];
        const double ref = 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
        const double err = std::abs((double)out[i] - ref);
        if (err > max_abs) { max_abs = err; worst = i; }
    }
    const bool pass = max_abs <= tol;
    printf("  erf_avx2_vs_f64   [%5.1f,%5.1f] n=%-7d  max_abs_err=%.2e  %s",
           lo, hi, n, max_abs, pass ? "PASS" : "FAIL");
    if (!pass) printf("  (worst x=%.4f)", (double)inp[worst]);
    printf("\n");
    return pass;
}

using GeluFn = void (*)(const float*, float*, std::size_t);

static bool test_gelu_special_values(GeluFn fn, const char* name) {
    const float special[] = {INFINITY, -INFINITY, NAN, 0.0f, -0.0f, 1e-40f, -1e-40f, -20.0f};
    const int ns = sizeof(special) / sizeof(special[0]);
    bool pass = true;
    for (int offset : {0, 9}) {  // 8-wide body, masked tail
        float in[32] = {}, out[32] = {};
        for (int i = 0; i < ns; ++i) in[offset + i] = special[i];
        fn(in, out, static_cast<std::size_t>(offset + ns));
        const float* r = out + offset;
        pass &= std::isinf(r[0]) && r[0] > 0;
        pass &= r[1] == 0.0f;                 // GeLU(-inf) = -0, not NaN or -huge
        pass &= std::isnan(r[2]);
        pass &= r[3] == 0.0f && r[4] == 0.0f;
        pass &= std::abs(r[5]) <= 1e-40f && std::abs(r[6]) <= 1e-40f;
        pass &= r[7] == 0.0f;                 // x < -10 returns -0
    }
    printf("  special values (inf, -inf, nan, ±0, subnormal, -20) %-5s  %s\n",
           name, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_gelu_tail_matches_body(GeluFn fn, const char* name) {
    float in[16], full[16], part[13];
    for (int i = 0; i < 16; ++i) in[i] = -4.0f + 0.53f * i;
    fn(in, full, 16);   // all 16 in the 8-wide body
    fn(in, part, 13);   // elements 8..12 in the masked tail
    bool pass = true;
    for (int i = 0; i < 13; ++i) pass &= std::memcmp(&part[i], &full[i], sizeof(float)) == 0;
    printf("  tail bitwise == body                    %-5s  %s\n", name, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_gelu_scalar_vs_erf_ref(int n, float lo, float hi,
                                          double tol = 1e-5) {
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    for (int i = 0; i < n; ++i)
        inp[i] = lo + (hi - lo) * i / std::max(n - 1, 1);

    gelu_forward_scalar(inp.get(), out.get(), n);

    double max_abs = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = gelu_erf_ref(inp[i]);
        double err = std::abs((double)out[i] - ref);
        max_abs = std::max(max_abs, err);
    }
    bool pass = (max_abs < tol);
    printf("  scalar_vs_erf_ref [%5.1f,%5.1f] n=%-7d  max_abs_err=%.2e  %s\n",
           lo, hi, n, max_abs, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_gelu_inplace(int n) {
    auto buf1 = make_aligned_array<float>(n);
    auto buf2 = make_aligned_array<float>(n);
    for (int i = 0; i < n; ++i)
        buf1[i] = buf2[i] = -5.0f + 10.0f * i / std::max(n - 1, 1);

    // Out-of-place
    auto out = make_aligned_array<float>(n);
    gelu_forward_avx2(buf1.get(), out.get(), n);

    // In-place on a copy
    gelu_forward_avx2(buf2.get(), buf2.get(), n);

    double max_diff = 0.0;
    for (int i = 0; i < n; ++i)
        max_diff = std::max(max_diff, std::abs((double)out[i] - (double)buf2[i]));

    bool pass = (max_diff == 0.0);
    printf("  inplace_vs_outofplace          n=%-7d  max_diff=%.2e   %s\n",
           n, max_diff, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_gelu_saturation() {
    // GeLU(large positive) ≈ x
    auto inp = make_aligned_array<float>(8);
    auto out = make_aligned_array<float>(8);
    for (int i = 0; i < 8; ++i) inp[i] = 15.0f;
    gelu_forward_avx2(inp.get(), out.get(), 8);
    bool pos_ok = true;
    for (int i = 0; i < 8; ++i)
        if (std::abs(out[i] - 15.0f) > 0.01f) { pos_ok = false; break; }

    // GeLU(large negative) ≈ 0
    for (int i = 0; i < 8; ++i) inp[i] = -15.0f;
    gelu_forward_avx2(inp.get(), out.get(), 8);
    bool neg_ok = true;
    for (int i = 0; i < 8; ++i)
        if (std::abs(out[i]) > 0.01f) { neg_ok = false; break; }

    // GeLU(0) == 0
    for (int i = 0; i < 8; ++i) inp[i] = 0.0f;
    gelu_forward_avx2(inp.get(), out.get(), 8);
    bool zero_ok = true;
    for (int i = 0; i < 8; ++i)
        if (std::abs(out[i]) > 1e-6f) { zero_ok = false; break; }

    bool pass = pos_ok && neg_ok && zero_ok;
    printf("  saturation_and_zero            GeLU(15)≈15=%s  GeLU(-15)≈0=%s  GeLU(0)=0=%s  %s\n",
           pos_ok?"Y":"N", neg_ok?"Y":"N", zero_ok?"Y":"N", pass?"PASS":"FAIL");
    return pass;
}

static bool test_gelu_no_nan_inf(int n) {
    // Verify no NaN/Inf across the full float32 exp-safe range
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    for (int i = 0; i < n; ++i)
        inp[i] = -88.0f + 176.0f * i / std::max(n - 1, 1);

    gelu_forward_avx2(inp.get(), out.get(), n);

    bool ok = true;
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(out[i])) { ok = false; break; }
    printf("  no_nan_inf   x∈[-88,88]        n=%-7d              %s\n",
           n, ok ? "PASS" : "FAIL");
    return ok;
}

int run_gelu_tests() {
    printf("\n── GeLU Correctness Tests ──\n");
    bool all = true;

    printf(" AVX2 fast path vs tanh-formula reference (should match to < 2e-5):\n");
    all &= test_gelu_avx2_vs_tanh_ref(1024,  -5.0f,  5.0f);
    all &= test_gelu_avx2_vs_tanh_ref(8192,  -3.0f,  3.0f);
    all &= test_gelu_avx2_vs_tanh_ref(512,   -0.5f,  0.5f);
    all &= test_gelu_avx2_vs_tanh_ref(256,    2.0f,  8.0f);
    all &= test_gelu_avx2_vs_tanh_ref(256,   -8.0f, -2.0f);
    all &= test_gelu_avx2_vs_tanh_ref(7,     -5.0f,  5.0f);   // tail
    all &= test_gelu_avx2_vs_tanh_ref(1,      0.0f,  0.0f);   // single

    printf(" AVX2 exact (erf) path vs float64 reference (bound 2e-6, 1e-6 on [-3,3]):\n");
    all &= test_gelu_erf_avx2_vs_ref(400001, -10.0f, 10.0f);
    all &= test_gelu_erf_avx2_vs_ref(60001,   -3.0f,  3.0f, 1e-6);
    all &= test_gelu_erf_avx2_vs_ref(7,       -5.0f,  5.0f);   // tail only

    printf(" Special values and tail consistency (both variants):\n");
    all &= test_gelu_special_values(gelu_forward_avx2, "tanh");
    all &= test_gelu_special_values(gelu_erf_forward_avx2, "erf");
    all &= test_gelu_tail_matches_body(gelu_forward_avx2, "tanh");
    all &= test_gelu_tail_matches_body(gelu_erf_forward_avx2, "erf");

    printf(" Scalar path vs erff reference (should match to < 1e-5):\n");
    all &= test_gelu_scalar_vs_erf_ref(1024,  -5.0f,  5.0f);
    all &= test_gelu_scalar_vs_erf_ref(512,   -3.0f,  3.0f);

    printf(" In-place vs out-of-place consistency:\n");
    all &= test_gelu_inplace(8);
    all &= test_gelu_inplace(1024);
    all &= test_gelu_inplace(65536);

    printf(" Saturation and boundary behaviour:\n");
    all &= test_gelu_saturation();
    all &= test_gelu_no_nan_inf(10000);

    printf("  Overall: %s\n", all ? "ALL PASS" : "SOME FAILURES");
    return all ? 0 : 1;
}
