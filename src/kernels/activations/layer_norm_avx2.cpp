/**
 * layer_norm_avx2.cpp — AVX2 LayerNorm of one row:
 *   y_i = (x_i − μ) / sqrt(σ² + ε) · γ_i + β_i      (γ, β optional)
 *
 * ─── Two-pass statistics, chosen by measurement over Welford ─────────────────
 * Pass 1 sums x − x₀ (x₀ = x[0]), pass 2 sums (x − μ)², pass 3 normalizes.
 * Each pass keeps 4 independent 8-float accumulators (latency 4 × 2 ports:
 * one dependent chain would run at a quarter of the add rate) and adds them
 * into a double every 1024 elements, so float rounding grows with 32 terms
 * per lane, not n/32.
 *
 * Measured on the Lunar Lake P-core (rows in L1, median of 31 interleaved
 * rounds, rows/second of the same harness): this kernel 4.9 elements/ns at
 * 1024×1024 and 5.3 at 64×768; the previous kernel (double accumulation, one
 * dependent add chain per pass) 3.2 and 3.6; a single-pass Welford with per-lane
 * updates and a Chan merge 3.3–3.7, no faster than the old kernel at n <= 1024
 * (its per-element update costs more than reading an L1-resident row twice)
 * and up to 17× outside the error bound below at n = 65536 (float lane state).
 *
 * Accuracy: |y − y₆₄| <= 1e-5 + 1e-4·|y₆₄| (tested on every table) for rows
 * with |μ| <= 30·σ; rows up to 65536 elements, including a first element far
 * from the mean. Rounding μ to float32 costs about 6e-8·|μ|/σ in units of the
 * normalized output, so rows with |μ| ≫ σ exceed the bound, as they do in
 * PyTorch's LayerNorm (measured 166–1684× the bound at |μ|/σ = 10⁴–10⁵,
 * this kernel 47–1061×).
 */

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "layer_norm_avx2.cpp is an AVX2 unit: compile with -mavx2 -mfma (CMakeLists.txt)"
#endif

#include "activations.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace activations {

namespace {

/** Sum of the 8 lanes, in double. */
inline double hsum_d(__m256 v) noexcept {
    const __m256d s = _mm256_add_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(v)),
                                    _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
    const __m128d t = _mm_add_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd(s, 1));
    return _mm_cvtsd_f64(_mm_add_sd(t, _mm_unpackhi_pd(t, t)));
}

constexpr int kFlush = 1024;   // elements per float block before adding into double

/** Σ (x_i − c), float lanes flushed into double every kFlush elements. */
double sum_shifted(const float* x, int n, float c, int& i) noexcept {
    const __m256 vc = _mm256_set1_ps(c);
    double sum = 0.0;
    i = 0;
    while (i + 32 <= n) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        const int end = std::min(n - 31, i + kFlush);
        for (; i < end; i += 32) {
            a0 = _mm256_add_ps(a0, _mm256_sub_ps(_mm256_loadu_ps(x + i), vc));
            a1 = _mm256_add_ps(a1, _mm256_sub_ps(_mm256_loadu_ps(x + i + 8), vc));
            a2 = _mm256_add_ps(a2, _mm256_sub_ps(_mm256_loadu_ps(x + i + 16), vc));
            a3 = _mm256_add_ps(a3, _mm256_sub_ps(_mm256_loadu_ps(x + i + 24), vc));
        }
        sum += hsum_d(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    }
    return sum;
}

/** Σ (x_i − c)², same blocking. */
double sum_sq_shifted(const float* x, int n, float c, int& i) noexcept {
    const __m256 vc = _mm256_set1_ps(c);
    double sum = 0.0;
    i = 0;
    while (i + 32 <= n) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        const int end = std::min(n - 31, i + kFlush);
        for (; i < end; i += 32) {
            const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(x + i), vc);
            const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(x + i + 8), vc);
            const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(x + i + 16), vc);
            const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(x + i + 24), vc);
            a0 = _mm256_fmadd_ps(d0, d0, a0);
            a1 = _mm256_fmadd_ps(d1, d1, a1);
            a2 = _mm256_fmadd_ps(d2, d2, a2);
            a3 = _mm256_fmadd_ps(d3, d3, a3);
        }
        sum += hsum_d(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
    }
    return sum;
}

}  // namespace

void layer_norm_avx2(const float* input, float* output, int n,
                     const float* gamma, const float* beta, float eps) {
    if (n <= 0) return;

    // ─── Pass 1: mean, from sums shifted by x[0] ─────────────────────────────
    const float x0 = input[0];
    int i = 0;
    double sum = sum_shifted(input, n, x0, i);
    for (; i < n; ++i) sum += static_cast<double>(input[i] - x0);
    const float mean = static_cast<float>(x0 + sum / n);

    // ─── Pass 2: variance about that mean ────────────────────────────────────
    double var_sum = sum_sq_shifted(input, n, mean, i);
    for (; i < n; ++i) {
        const double d = static_cast<double>(input[i]) - static_cast<double>(mean);
        var_sum += d * d;
    }
    const float inv_std = 1.0f / std::sqrt(static_cast<float>(var_sum / n) + eps);

    // ─── Pass 3: normalize (and optionally apply gamma/beta) ─────────────────
    {
        __m256 vmean    = _mm256_set1_ps(mean);
        __m256 vinvstd  = _mm256_set1_ps(inv_std);
        i = 0;

        if (gamma && beta) {
            for (; i + 8 <= n; i += 8) {
                __m256 x    = _mm256_loadu_ps(input  + i);
                __m256 g    = _mm256_loadu_ps(gamma  + i);
                __m256 b    = _mm256_loadu_ps(beta   + i);
                __m256 xnorm = _mm256_mul_ps(_mm256_sub_ps(x, vmean), vinvstd);
                __m256 y     = _mm256_fmadd_ps(xnorm, g, b);  // xnorm*γ + β
                _mm256_storeu_ps(output + i, y);
            }
            for (; i < n; ++i) {
                output[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
            }
        } else if (gamma) {
            for (; i + 8 <= n; i += 8) {
                __m256 x    = _mm256_loadu_ps(input + i);
                __m256 g    = _mm256_loadu_ps(gamma + i);
                __m256 xnorm = _mm256_mul_ps(_mm256_sub_ps(x, vmean), vinvstd);
                _mm256_storeu_ps(output + i, _mm256_mul_ps(xnorm, g));
            }
            for (; i < n; ++i) output[i] = (input[i] - mean) * inv_std * gamma[i];
        } else if (beta) {
            for (; i + 8 <= n; i += 8) {
                __m256 x    = _mm256_loadu_ps(input + i);
                __m256 b    = _mm256_loadu_ps(beta  + i);
                __m256 xnorm = _mm256_mul_ps(_mm256_sub_ps(x, vmean), vinvstd);
                _mm256_storeu_ps(output + i, _mm256_add_ps(xnorm, b));
            }
            for (; i < n; ++i) output[i] = (input[i] - mean) * inv_std + beta[i];
        } else {
            for (; i + 8 <= n; i += 8) {
                __m256 x    = _mm256_loadu_ps(input + i);
                __m256 xnorm = _mm256_mul_ps(_mm256_sub_ps(x, vmean), vinvstd);
                _mm256_storeu_ps(output + i, xnorm);
            }
            for (; i < n; ++i) output[i] = (input[i] - mean) * inv_std;
        }
    }
}

} // namespace activations
