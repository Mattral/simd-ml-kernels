/**
 * silu_avx2.cpp — AVX2 SiLU (Swish): SiLU(x) = x · σ(x) = x / (1 + exp(−x)).
 *
 * Why x / (1 + exp(−x)) and not x · 0.5·(1 + tanh(x/2)), the form used before:
 *   - tanh_avx2 clamps its argument at ±7.99, so for x < −16 the old form
 *     returned x · 1.1e-7 instead of x · exp(x): 2e-6 at x = −20, 1e-4 at
 *     x = −1000, 5e31 at x = −FLT_MAX. Even unclamped, 1 + tanh(x/2) cancels
 *     catastrophically for x ≪ 0 (relative error 1.4e-3 at x = −10).
 *   - exp(−x) is accurate in relative terms over the whole range, and the
 *     division is correctly rounded, so the result has a relative error bound.
 *   - Measured on the Lunar Lake P-core (Lion Cove, 4096 floats in L1, min of
 *     31 interleaved rounds): 0.29 ns/element against 0.43 for the tanh form
 *     and 0.30 for rcp + one Newton step (whose error is 6x larger). VDIVPS
 *     runs on its own divider, overlapping the ~16 FMA/ALU uops of exp.
 *
 * Range: fast_exp_avx2 clamps its argument to ±88.38, so for x < −88 the
 * denominator would stay finite (2.4e38) and x / d would grow with |x|
 * (−1.4 at x = −FLT_MAX). For −x > 88 the denominator is set to +inf instead:
 * the result is −0 (true |SiLU(x)| < 88·e^−88 = 1.5e-36) and −inf / inf = NaN,
 * which is what the scalar kernel and PyTorch return for SiLU(−inf).
 *
 * Error, every finite float (2^32 − 2^24 values, AVX2 table, this file):
 *   |err| ≤ 1e-35 + 2.9e-7 · |SiLU(x)|, max abs error 1.3e-6 (at x ≈ 13.8).
 * The tests assert 1e-35 + 1e-6 · |ref| for every table.
 */

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "silu_avx2.cpp is an AVX2 unit: compile with -mavx2 -mfma (CMakeLists.txt)"
#endif

#include "activations.hpp"
#include <immintrin.h>
#include <cmath>
#include "../simd_math.hpp"

namespace activations {

static inline __m256 silu8(__m256 x) {
    const __m256 neg_x = _mm256_xor_ps(x, _mm256_set1_ps(-0.0f));
    __m256 d = _mm256_add_ps(_mm256_set1_ps(1.0f), fast_exp_avx2(neg_x));
    // NaN compares false, so NaN keeps a finite d and x / d stays NaN.
    const __m256 huge = _mm256_cmp_ps(neg_x, _mm256_set1_ps(88.0f), _CMP_GT_OQ);
    d = _mm256_blendv_ps(d, _mm256_set1_ps(INFINITY), huge);
    return _mm256_div_ps(x, d);
}

void silu_avx2(const float* input, float* output, int n) {
    if (n <= 0) return;
    apply_elementwise_avx2(input, output, static_cast<std::size_t>(n), silu8);
}

} // namespace activations
