/**
 * relu_avx2.cpp — AVX2-vectorized ReLU activation function.
 *
 * ReLU(x) = max(0, x)
 *
 * Implementation: _mm256_max_ps with a zero vector eliminates all branches.
 * Processes 8 floats per instruction at 0.33 cycle throughput on Skylake.
 */

#include "../cache_alloc.hpp"
#if !defined(__AVX2__) || !defined(__FMA__)
#  error "relu_avx2.cpp is an AVX2 unit: compile with -mavx2 -mfma (CMakeLists.txt)"
#endif

#include "activations.hpp"
#include <immintrin.h>
#include <cstddef>

namespace activations {

void relu_avx2(const float* input, float* output, int n) {
    int i = 0;
    const int simd_width = 8;
    const __m256 zero = _mm256_setzero_ps();

    // Main vectorized loop: 8 elements per iteration
    for (; i + simd_width <= n; i += simd_width) {
        __m256 v = _mm256_loadu_ps(input + i);
        __m256 r = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(output + i, r);
    }
    // Scalar tail
    for (; i < n; ++i) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

} // namespace activations
