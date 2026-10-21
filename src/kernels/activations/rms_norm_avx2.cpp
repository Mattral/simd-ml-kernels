/**
 * rms_norm_avx2.cpp — AVX2 RMSNorm of one row (Llama's LlamaRMSNorm):
 *   y_i = w_i · (x_i · r),   r = 1 / sqrt(Σ x² / n + ε)      (w optional)
 *
 * Why a kernel for three lines of NumPy: in a decode step it runs right after
 * a GEMV has streamed megabytes of weights through the caches, and each of
 * the four NumPy operations it took (x*x, mean, two multiplies) then starts
 * cold. Dev runs (Lunar Lake P-core, Llama-3.2-1B Q4_0, 4 threads): the NumPy
 * form took 3.9 µs alone and 25 µs per call inside the decode step (33 calls,
 * 0.82 ms per token); a single C++ call of the same size after a GEMV took
 * 2.0 µs against 6.7 for the NumPy form.
 *
 * Σ x² keeps 4 independent 8-float FMA accumulators (as layer_norm_avx2.cpp)
 * added into a double every 1024 elements, so float rounding grows with 32
 * terms per lane, not n/8. r = 1 / sqrt(mean + ε), the square root and the
 * division in double (no rsqrt estimate, whatever the compiler; clang-15 used
 * one for the float form and returned NaN for an infinite row), rounded to
 * float, and the products in transformers' order: x·r first, then w·(x·r).
 *
 * Accuracy against float64: |y − y₆₄| <= 1e-30 + 1e-6·|y₆₄| (tested on every
 * table; mean and r each carry a few 2^-24 relative, the two products one
 * rounding each). A NaN anywhere in x makes the whole row NaN; ±inf makes
 * Σx² = inf, r = 0, so finite x give 0 and the inf gives NaN (as PyTorch).
 */

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "rms_norm_avx2.cpp is an AVX2 unit: compile with -mavx2 -mfma (CMakeLists.txt)"
#endif

#include "activations.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace activations {

namespace {

inline double hsum_d(__m256 v) noexcept {
    const __m256d s = _mm256_add_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(v)),
                                    _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
    const __m128d t = _mm_add_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd(s, 1));
    return _mm_cvtsd_f64(_mm_add_sd(t, _mm_unpackhi_pd(t, t)));
}

constexpr int kFlush = 1024;   // elements per float block before adding into double

}  // namespace

void rms_norm_avx2(const float* input, float* output, int n, const float* weight, float eps) {
    if (n <= 0) return;
    double sum = 0.0;
    int i = 0;
    while (i + 32 <= n) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        const int end = std::min(n - 31, i + kFlush);
        for (; i < end; i += 32) {
            const __m256 x0 = _mm256_loadu_ps(input + i);
            const __m256 x1 = _mm256_loadu_ps(input + i + 8);
            const __m256 x2 = _mm256_loadu_ps(input + i + 16);
            const __m256 x3 = _mm256_loadu_ps(input + i + 24);
            a0 = _mm256_fmadd_ps(x0, x0, a0);
            a1 = _mm256_fmadd_ps(x1, x1, a1);
            a2 = _mm256_fmadd_ps(x2, x2, a2);
            a3 = _mm256_fmadd_ps(x3, x3, a3);
        }
        sum += hsum_d(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    }
    for (; i < n; ++i) sum += static_cast<double>(input[i]) * input[i];
    // In double: under -ffast-math clang turns a float 1/sqrtf into vrsqrtss
    // plus a Newton step (not correctly rounded; NaN for an infinite sum).
    // AVX2 has no double rsqrt estimate, so this stays sqrtsd + divsd.
    const float r = static_cast<float>(1.0 / std::sqrt(static_cast<double>(static_cast<float>(sum / n) + eps)));

    const __m256 vr = _mm256_set1_ps(r);
    i = 0;
    if (weight) {
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(weight + i),
                                                       _mm256_mul_ps(_mm256_loadu_ps(input + i), vr)));
        for (; i < n; ++i) output[i] = weight[i] * (input[i] * r);
    } else {
        for (; i + 8 <= n; i += 8)
            _mm256_storeu_ps(output + i, _mm256_mul_ps(_mm256_loadu_ps(input + i), vr));
        for (; i < n; ++i) output[i] = input[i] * r;
    }
}

}  // namespace activations
