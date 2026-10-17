/**
 * layer_norm_avx2.cpp — AVX2-vectorized Layer Normalization.
 *
 * LayerNorm(x)_i = (x_i - μ) / sqrt(σ² + ε) * γ_i + β_i
 *
 * where:
 *   μ  = (1/n) * Σ x_i              (mean)
 *   σ² = (1/n) * Σ (x_i - μ)²       (variance)
 *   γ_i, β_i                         (learnable scale/shift, optional)
 *   ε                                 (numerical stability constant, default 1e-5)
 *
 * ─── Micro-Architectural Notes ───────────────────────────────────────────────
 * LayerNorm is memory-bound for large n (two passes over the data). For small n
 * (e.g., 512 or 768 as in BERT), the data fits in L1/L2 cache and throughput
 * is dominated by the FMA units.
 *
 * The two-pass approach (pass 1: mean; pass 2: variance + normalize) is
 * preferred over the single-pass Welford algorithm here because it produces
 * simpler, more pipeline-friendly AVX2 code. The single-pass approach would
 * reduce memory traffic for very large n — noted in ROADMAP.md.
 *
 * ─── Vectorization Strategy ──────────────────────────────────────────────────
 * - Use double-precision horizontal accumulation for the mean and variance sums
 *   to avoid catastrophic cancellation when inputs have similar magnitudes.
 * - Use AVX2 _mm256_hadd_ps + scalar reduction for the final summation.
 * - The normalize pass uses AVX2 for full throughput.
 */

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "layer_norm_avx2.cpp is an AVX2 unit: compile with -mavx2 -mfma (CMakeLists.txt)"
#endif

#include "activations.hpp"
#include <immintrin.h>
#include <cmath>
#include <cstddef>

namespace activations {

void layer_norm_avx2(const float* input, float* output, int n,
                     const float* gamma, const float* beta, float eps) {
    if (n <= 0) return;

    // ─── Pass 1: Compute mean ────────────────────────────────────────────────
    // Accumulate in double to avoid precision loss for large n.
    double sum = 0.0;

    {
        __m256d acc = _mm256_setzero_pd();
        int i = 0;
        // Process 8 floats at a time: convert float→double 4 at a time
        for (; i + 8 <= n; i += 8) {
            __m256 v8   = _mm256_loadu_ps(input + i);
            // Split into two groups of 4 for float→double conversion
            __m128 lo4  = _mm256_castps256_ps128(v8);
            __m128 hi4  = _mm256_extractf128_ps(v8, 1);
            acc = _mm256_add_pd(acc, _mm256_cvtps_pd(lo4));
            acc = _mm256_add_pd(acc, _mm256_cvtps_pd(hi4));
        }
        // Horizontal sum of the four double accumulators
        double tmp[4];
        _mm256_storeu_pd(tmp, acc);
        sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        // Scalar tail
        for (; i < n; ++i) sum += static_cast<double>(input[i]);
    }

    const float mean = static_cast<float>(sum / n);

    // ─── Pass 2: Compute variance ────────────────────────────────────────────
    double var_sum = 0.0;

    {
        __m256d vacc = _mm256_setzero_pd();
        __m256  vmean = _mm256_set1_ps(mean);
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 v8   = _mm256_loadu_ps(input + i);
            __m256 diff = _mm256_sub_ps(v8, vmean);           // (x - mean)
            __m256 sq   = _mm256_mul_ps(diff, diff);           // (x - mean)²
            __m128 lo4  = _mm256_castps256_ps128(sq);
            __m128 hi4  = _mm256_extractf128_ps(sq, 1);
            vacc = _mm256_add_pd(vacc, _mm256_cvtps_pd(lo4));
            vacc = _mm256_add_pd(vacc, _mm256_cvtps_pd(hi4));
        }
        double vtmp[4];
        _mm256_storeu_pd(vtmp, vacc);
        var_sum = vtmp[0] + vtmp[1] + vtmp[2] + vtmp[3];
        for (; i < n; ++i) {
            double d  = static_cast<double>(input[i]) - static_cast<double>(mean);
            var_sum  += d * d;
        }
    }

    const float inv_std = 1.0f / std::sqrt(static_cast<float>(var_sum / n) + eps);

    // ─── Pass 3: Normalize (and optionally apply gamma/beta) ─────────────────
    {
        __m256 vmean    = _mm256_set1_ps(mean);
        __m256 vinvstd  = _mm256_set1_ps(inv_std);
        int i = 0;

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
