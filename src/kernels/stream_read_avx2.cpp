/**
 * stream_read_avx2.cpp — the GEMV-pattern read streams of stream_read.hpp.
 * An AVX2 unit. 4 rows keep 2 YMM accumulators each, 8 rows one each (the
 * 16 YMM registers), so at least 8 independent add chains: the adds never
 * bound the loop, which measures how fast this access pattern reads.
 */

#include "stream_read.hpp"

#include <immintrin.h>

#if !defined(_MSC_VER) && !defined(__AVX2__)
#  error "stream_read_avx2.cpp is an AVX2 unit (CMakeLists.txt)"
#endif

namespace simd_ml {

namespace {

template <int R, bool Prefetch>
double rows_sum(const float* p, std::ptrdiff_t groups, std::ptrdiff_t L) noexcept {
    constexpr int A = R <= 4 ? 2 : 1;   // accumulators per row
    __m256 acc[R][A];
    for (auto& r : acc)
        for (auto& v : r) v = _mm256_setzero_ps();
    for (std::ptrdiff_t g = 0; g < groups; ++g) {
        const float* base = p + g * R * L;
        for (std::ptrdiff_t k = 0; k + 16 <= L; k += 16)
            for (int r = 0; r < R; ++r) {
                const float* q = base + r * L + k;
                if (Prefetch) _mm_prefetch(reinterpret_cast<const char*>(q + R * L), _MM_HINT_T0);
                acc[r][0] = _mm256_add_ps(acc[r][0], _mm256_loadu_ps(q));
                acc[r][A - 1] = _mm256_add_ps(acc[r][A - 1], _mm256_loadu_ps(q + 8));
            }
    }
    __m256 s = _mm256_setzero_ps();
    for (auto& r : acc)
        for (int v = 0; v < A; ++v) s = _mm256_add_ps(s, r[v]);
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, s);
    double total = 0.0;
    for (float v : lanes) total += v;
    return total;
}

}  // namespace

double stream_read_rows_avx2(const float* p, std::ptrdiff_t n, int rows, std::ptrdiff_t row_floats,
                             bool prefetch) noexcept {
    const std::ptrdiff_t L = row_floats / 16 * 16;
    if (L <= 0 || (rows != 4 && rows != 8)) return 0.0;
    const std::ptrdiff_t groups = n / (rows * L);
    double total = 0.0;
    if (rows == 4) total = prefetch ? rows_sum<4, true>(p, groups, L) : rows_sum<4, false>(p, groups, L);
    else total = prefetch ? rows_sum<8, true>(p, groups, L) : rows_sum<8, false>(p, groups, L);
    for (std::ptrdiff_t i = groups * rows * L; i < n; ++i) total += p[i];   // rest
    return total;
}

}  // namespace simd_ml
