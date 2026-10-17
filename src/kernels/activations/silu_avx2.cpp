/**
 * silu_avx2.cpp — AVX2-vectorized SiLU (Swish) activation function.
 *
 * SiLU(x) = x · σ(x)  where σ(x) = 1 / (1 + exp(−x))
 *
 * Implementation:
 *   σ(x) = 0.5 · (1 + tanh(x/2))
 *
 * We use the production-grade Cody–Waite exp-based tanh from simd_math.hpp.
 * This gives max absolute error < 2e-7 vs float32 reference sigmoid.
 *
 * Note on the tanh identity: σ(x) = 0.5·(1 + tanh(x/2)) is mathematically
 * exact (derivable from the definition of tanh via exp) and numerically
 * stable for all finite x. It avoids exp(−x) overflow for large positive x.
 */

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "silu_avx2.cpp is an AVX2 unit: compile with -mavx2 -mfma (CMakeLists.txt)"
#endif

#include "activations.hpp"
#include <immintrin.h>
#include "../simd_math.hpp"
#include <cmath>
#include <cstddef>


namespace activations {

void silu_avx2(const float* input, float* output, int n) {
    const __m256 half = _mm256_set1_ps(0.5f);

    int i = 0;
    // Main AVX2 loop: 8 elements per iteration
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);

        // σ(x) = 0.5 * (1 + tanh(x/2))
        // tanh_avx2 has max abs error < 2e-7 vs float32 reference tanh
        __m256 half_x = _mm256_mul_ps(x, half);
        __m256 t      = tanh_avx2(half_x);             // tanh(x/2)
        __m256 sig    = _mm256_fmadd_ps(half, t, half); // 0.5*t + 0.5 = 0.5*(1+t)

        // SiLU(x) = x * σ(x)
        _mm256_storeu_ps(output + i, _mm256_mul_ps(x, sig));
    }
    // Scalar tail
    for (; i < n; ++i) {
        float x     = input[i];
        float sig   = 1.0f / (1.0f + std::exp(-x));
        output[i]   = x * sig;
    }
}

} // namespace activations
