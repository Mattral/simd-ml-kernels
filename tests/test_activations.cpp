/**
 * test_activations.cpp — Correctness tests for activation function kernels.
 *
 * Tests: ReLU, SiLU, Softmax, LayerNorm — all validated against scalar
 * reference implementations with tight numerical tolerance.
 */

#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include "../src/kernels/cache_alloc.hpp"
#include "../src/kernels/activations/activations.hpp"
#include "kernel_tables.hpp"

// The table the test functions below call into (set by the runner).
static const simd_ml::KernelTable* T = nullptr;

static void fill_sweep(float* buf, int n, float lo, float hi) {
    for (int i = 0; i < n; ++i)
        buf[i] = lo + (hi - lo) * i / std::max(n - 1, 1);
}

static void fill_random(float* buf, int n, unsigned seed = 42) {
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (float)(int(s >> 16) % 200 - 100) * 0.05f;
    }
}

// ─── ReLU tests ──────────────────────────────────────────────────────────────
static bool test_relu(int n, float lo = -5.0f, float hi = 5.0f) {
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    fill_sweep(inp.get(), n, lo, hi);

    T->relu(inp.get(), out.get(), n);

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        float ref = inp[i] > 0.0f ? inp[i] : 0.0f;
        double err = std::abs((double)out[i] - (double)ref);
        max_err = std::max(max_err, err);
    }
    bool pass = (max_err == 0.0);
    printf("  ReLU n=%-7d  max_abs_err=%.2e  %s\n", n, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ─── SiLU tests ──────────────────────────────────────────────────────────────
// Bound from silu_avx2.cpp: |err| <= 1e-35 + 1e-6*|ref| over every finite
// float (measured 2.9e-7 relative). Returns the worst err / bound (pass: <= 1).
static double silu_worst(const float* inp, const float* out, int n) {
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
        const double x = inp[i], ref = x / (1.0 + std::exp(-x));
        worst = std::max(worst, std::abs((double)out[i] - ref) / (1e-35 + 1e-6 * std::abs(ref)));
    }
    return worst;
}

static bool test_silu(int n, float lo, float hi) {
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    fill_sweep(inp.get(), n, lo, hi);

    T->silu(inp.get(), out.get(), n);

    const double worst = silu_worst(inp.get(), out.get(), n);
    bool pass = worst <= 1.0;
    printf("  SiLU n=%-7d [%g, %g]  max err/bound=%.2f  %s\n", n, lo, hi, worst,
           pass ? "PASS" : "FAIL");
    return pass;
}

// ±m·2^e for every binade e in [-126, 127]: the old AVX2 kernel was wrong
// only for x < -16 (error 1e-4 at -1000, 5e31 at -FLT_MAX).
static bool test_silu_magnitudes() {
    const int n = 2 * 254 * 3 + 5;  // + 5: the last 5 go through the masked tail
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    int k = 0;
    for (int e = -126; e <= 127; ++e)
        for (float m : {1.0f, 1.3f, 1.9999999f})
            for (float s : {1.0f, -1.0f}) inp[k++] = s * std::ldexp(m, e);
    for (float x : {-88.5f, -88.0f, -87.5f, 88.5f, -103.0f}) inp[k++] = x;
    T->silu(inp.get(), out.get(), n);
    const double worst = silu_worst(inp.get(), out.get(), n);
    bool pass = worst <= 1.0;
    printf("  SiLU every binade 2^-126..2^127  max err/bound=%.2f  %s\n", worst,
           pass ? "PASS" : "FAIL");
    return pass;
}

// SiLU(+inf) = +inf; SiLU(-inf) = NaN (-inf / inf, as the scalar kernel and
// PyTorch); NaN stays NaN; signed zeros pass through. Offsets 0 and 9 put the
// values in the 8-wide body and in the masked tail.
static bool test_silu_special() {
    bool pass = true;
    for (int offset : {0, 9}) {
        const float special[] = {INFINITY, -INFINITY, NAN, 0.0f, -0.0f};
        const int n = offset + 5;
        auto inp = make_aligned_array<float>(n);
        auto out = make_aligned_array<float>(n);
        for (int i = 0; i < offset; ++i) inp[i] = 1.0f;
        for (int i = 0; i < 5; ++i) inp[offset + i] = special[i];
        T->silu(inp.get(), out.get(), n);
        const float* o = out.get() + offset;
        pass &= o[0] == INFINITY && std::isnan(o[1]) && std::isnan(o[2]) &&
                o[3] == 0.0f && !std::signbit(o[3]) && o[4] == 0.0f;
    }
    printf("  SiLU +-inf, NaN, +-0 (body and tail)  %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ─── Softmax tests ────────────────────────────────────────────────────────────
static bool test_softmax(int n, unsigned seed = 1) {
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    fill_random(inp.get(), n, seed);

    T->softmax_row(inp.get(), out.get(), n);

    // Test 1: output sums to 1.0 (within FP32 tolerance)
    double total = 0.0;
    for (int i = 0; i < n; ++i) total += out[i];
    bool sum_ok = std::abs(total - 1.0) < 1e-5;

    // Test 2: all values in [0, 1]
    bool range_ok = true;
    for (int i = 0; i < n; ++i)
        if (out[i] < 0.0f || out[i] > 1.0f + 1e-6f) { range_ok = false; break; }

    // Test 3: argmax of softmax == argmax of input
    int argmax_in  = (int)(std::max_element(inp.get(), inp.get() + n) - inp.get());
    int argmax_out = (int)(std::max_element(out.get(), out.get() + n) - out.get());
    bool argmax_ok = (argmax_in == argmax_out);

    bool pass = sum_ok && range_ok && argmax_ok;
    printf("  Softmax n=%-6d  sum=%.6f  range_ok=%s  argmax_ok=%s  %s\n",
           n, total,
           range_ok  ? "YES" : "NO",
           argmax_ok ? "YES" : "NO",
           pass ? "PASS" : "FAIL");
    return pass;
}

// Rows that torch.softmax maps to all-NaN: every element -inf, any NaN, any
// +inf. n = 4 is scalar-tail only, 8 vector body only, 11 both.
static bool test_softmax_nan_rows() {
    bool pass = true;
    for (int n : {4, 8, 11}) {
        for (int kind = 0; kind < 3; ++kind) {
            float in[16], out[16];
            for (int i = 0; i < n; ++i) in[i] = kind == 0 ? -INFINITY : 0.1f * i;
            if (kind == 1) in[n / 2] = NAN;
            if (kind == 2) in[n / 2] = INFINITY;
            T->softmax_row(in, out, n);
            for (int i = 0; i < n; ++i) pass &= std::isnan(out[i]);
        }
    }
    printf("  Softmax all -inf / NaN / +inf rows -> NaN   %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ─── LayerNorm tests ─────────────────────────────────────────────────────────
// Each of the four affine combinations (none / gamma / beta / both) is checked
// against a two-pass float64 reference with |out - ref| <= atol + rtol*|ref|,
// atol = 1e-5, rtol = 1e-4 — the same bound as TestLayerNorm in
// test_precision.py. Inputs have mean ~50 so the mean subtraction is exercised.
static bool test_layer_norm(int n, bool use_gamma, bool use_beta, unsigned seed = 5) {
    auto inp   = make_aligned_array<float>(n);
    auto out   = make_aligned_array<float>(n);
    auto gamma = make_aligned_array<float>(n);
    auto beta  = make_aligned_array<float>(n);

    fill_random(inp.get(), n, seed);
    for (int i = 0; i < n; ++i) inp[i] = 20.0f * inp[i] + 50.0f;
    // gamma ~ [0.5, 1.5], beta ~ [-1, 1]: a dropped beta changes outputs by O(1)
    for (int i = 0; i < n; ++i) {
        gamma[i] = 1.0f + 0.5f * ((float)(i % 10) / 10.0f - 0.5f);
        beta[i]  = 2.0f * ((float)(i % 7) / 7.0f - 0.5f);
    }

    const float* g = use_gamma ? gamma.get() : nullptr;
    const float* b = use_beta  ? beta.get()  : nullptr;

    T->layer_norm_row(inp.get(), out.get(), n, g, b, 1e-5f);

    double ref_mean = 0.0;
    for (int i = 0; i < n; ++i) ref_mean += inp[i];
    ref_mean /= n;
    double ref_var = 0.0;
    for (int i = 0; i < n; ++i) ref_var += (inp[i] - ref_mean) * (inp[i] - ref_mean);
    ref_var /= n;
    const double ref_inv_std = 1.0 / std::sqrt(ref_var + 1e-5);

    const double atol = 1e-5, rtol = 1e-4;
    double max_abs_err = 0.0;
    bool pass = true;
    for (int i = 0; i < n; ++i) {
        double ref = (inp[i] - ref_mean) * ref_inv_std;
        if (use_gamma) ref *= gamma[i];
        if (use_beta)  ref += beta[i];
        const double err = std::abs((double)out[i] - ref);
        max_abs_err = std::max(max_abs_err, err);
        if (err > atol + rtol * std::abs(ref)) pass = false;
    }
    const char* label = use_gamma ? (use_beta ? "gamma+beta" : "gamma") : (use_beta ? "beta" : "none");
    printf("  LayerNorm n=%-5d %-10s  max_abs_err=%.2e  %s\n",
           n, label, max_abs_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ─── Test runner ─────────────────────────────────────────────────────────────
// Rows the documented bound covers (|mean| <= 30*std) at lengths where float
// lane sums would drift (65536), and a first element far from the mean (pass 1
// sums x - x[0]). Same bound as test_layer_norm: 1e-5 + 1e-4*|ref| vs double.
static bool test_layer_norm_conditioning() {
    bool all = true;
    struct R { float mean, stdv; bool outlier_first; const char* what; };
    for (int n : {1027, 4096, 65536})
        for (R r : {R{30.0f, 1.0f, false, "mean 30 std 1"}, R{-3e4f, 1e3f, false, "mean -3e4 std 1e3"},
                    R{0.3f, 1e-2f, false, "mean 0.3 std 1e-2"}, R{0.0f, 1.0f, true, "x[0] = 1e3"}}) {
            auto inp = make_aligned_array<float>(n), out = make_aligned_array<float>(n);
            auto gamma = make_aligned_array<float>(n), beta = make_aligned_array<float>(n);
            unsigned s = 12345u + static_cast<unsigned>(n);
            auto unif = [&s]() { s = s * 1664525u + 1013904223u; return (s >> 8) * (1.0f / 16777216.0f); };
            for (int i = 0; i < n; ++i) {
                const float z = (unif() + unif() + unif() + unif() - 2.0f) * 1.7320508f;  // ~N(0,1)
                inp[i] = r.mean + r.stdv * z;
                gamma[i] = 1.0f + 2.0f * (unif() - 0.5f);
                beta[i] = 2.0f * (unif() - 0.5f);
            }
            if (r.outlier_first) inp[0] = 1e3f;
            T->layer_norm_row(inp.get(), out.get(), n, gamma.get(), beta.get(), 1e-5f);
            double mu = 0.0;
            for (int i = 0; i < n; ++i) mu += inp[i];
            mu /= n;
            double var = 0.0;
            for (int i = 0; i < n; ++i) var += (inp[i] - mu) * (inp[i] - mu);
            const double is = 1.0 / std::sqrt(var / n + 1e-5);
            double worst = 0.0;
            for (int i = 0; i < n; ++i) {
                const double ref = (inp[i] - mu) * is * gamma[i] + beta[i];
                worst = std::max(worst, std::abs(out[i] - ref) / (1e-5 + 1e-4 * std::abs(ref)));
            }
            const bool pass = worst <= 1.0;
            printf("  LayerNorm n=%-6d %-18s err/bound=%.3f  %s\n", n, r.what, worst, pass ? "PASS" : "FAIL");
            all &= pass;
        }
    return all;
}

int run_activation_tests() {
    printf("\n── Activation Correctness Tests ──\n");
    const bool all_pass = for_each_kernel_table([](const simd_ml::KernelTable& t) {
        T = &t;
        bool ok = true;

        printf(" ReLU:\n");
        for (int n : {1, 7, 8, 9, 64, 256, 1024, 65536})
            ok &= test_relu(n);

        printf(" SiLU:\n");
        for (int n : {1, 7, 8, 9, 64, 256, 1024, 65536}) {
            ok &= test_silu(n, -10.0f, 10.0f);
            ok &= test_silu(n, -200.0f, 200.0f);   // crosses the -88 cut-off
        }
        ok &= test_silu(65537, -1.7e38f, 1.7e38f);  // hi - lo must not overflow
        ok &= test_silu_magnitudes();
        ok &= test_silu_special();

        printf(" Softmax:\n");
        for (int n : {1, 2, 8, 16, 64, 512, 4096})
            ok &= test_softmax(n);
        ok &= test_softmax_nan_rows();

        printf(" LayerNorm:\n");
        // Sizes cover the scalar tail alone (1, 7), exact multiples of the 8-wide
        // AVX2 body (8, 16, 64, 768) and body + tail (9, 15, 17, 1027).
        for (int n : {1, 7, 8, 9, 15, 16, 17, 64, 768, 1027})
            for (bool use_gamma : {false, true})
                for (bool use_beta : {false, true})
                    ok &= test_layer_norm(n, use_gamma, use_beta);
        ok &= test_layer_norm_conditioning();
        return ok;
    });

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
