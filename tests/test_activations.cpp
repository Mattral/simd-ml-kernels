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

    activations::relu_avx2(inp.get(), out.get(), n);

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
static float ref_silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static bool test_silu(int n, float lo = -5.0f, float hi = 5.0f, double tol = 1e-4) {
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    fill_sweep(inp.get(), n, lo, hi);

    activations::silu_avx2(inp.get(), out.get(), n);

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = ref_silu(inp[i]);
        double err = std::abs((double)out[i] - ref) / (std::abs(ref) + 1e-7);
        max_err = std::max(max_err, err);
    }
    bool pass = (max_err < tol);
    printf("  SiLU n=%-7d  max_rel_err=%.2e  %s\n", n, max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// ─── Softmax tests ────────────────────────────────────────────────────────────
static bool test_softmax(int n, unsigned seed = 1) {
    auto inp = make_aligned_array<float>(n);
    auto out = make_aligned_array<float>(n);
    fill_random(inp.get(), n, seed);

    activations::softmax_row_avx2(inp.get(), out.get(), n);

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
            activations::softmax_row_avx2(in, out, n);
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

    activations::layer_norm_avx2(inp.get(), out.get(), n, g, b, 1e-5f);

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
int run_activation_tests() {
    printf("\n── Activation Correctness Tests ──\n");
    bool all_pass = true;

    printf(" ReLU:\n");
    for (int n : {1, 7, 8, 9, 64, 256, 1024, 65536})
        all_pass &= test_relu(n);

    printf(" SiLU:\n");
    for (int n : {1, 7, 8, 9, 64, 256, 1024, 65536})
        all_pass &= test_silu(n);

    printf(" Softmax:\n");
    for (int n : {1, 2, 8, 16, 64, 512, 4096})
        all_pass &= test_softmax(n);
    all_pass &= test_softmax_nan_rows();

    printf(" LayerNorm:\n");
    // Sizes cover the scalar tail alone (1, 7), exact multiples of the 8-wide
    // AVX2 body (8, 16, 64, 768) and body + tail (9, 15, 17, 1027).
    for (int n : {1, 7, 8, 9, 15, 16, 17, 64, 768, 1027})
        for (bool use_gamma : {false, true})
            for (bool use_beta : {false, true})
                all_pass &= test_layer_norm(n, use_gamma, use_beta);

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
