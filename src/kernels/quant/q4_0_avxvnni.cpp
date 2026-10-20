/**
 * q4_0_avxvnni.cpp — the Q4_0 GEMV inner loop with AVX-VNNI (VEX vpdpbusd on
 * YMM: Alder Lake, Lunar Lake, Sapphire Rapids and later). An AVX-VNNI unit;
 * gemv_q4_0_avx2 calls it after CPUID reported AVX-VNNI. One vpdpbusd per
 * 32 weights, accumulating onto the −8·Σx start (q4_0_rows.hpp).
 */

#include "q4_0.hpp"
#include "q4_0_rows.hpp"

#if !defined(_MSC_VER) && (!defined(__AVX2__) || !defined(__AVXVNNI__))
#  error "q4_0_avxvnni.cpp is an AVX-VNNI unit: compile with -mavx2 -mfma -mavxvnni (CMakeLists.txt)"
#endif

namespace simd_ml {
namespace quant {

namespace {

struct DotAvxVnni {
    static SIMD_ML_Q4_INLINE __m256i dot_acc(__m256i acc, __m256i u, __m256i s) noexcept {
        return _mm256_dpbusd_avx_epi32(acc, u, s);
    }
};

// Rows walked together and prefetch spacing (blocks between prefetches of
// the next row group): DESIGN.md §2.9 for the dev measurements.
constexpr int kRowsAvxVnni = 8;
constexpr int kPrefetchAvxVnni = 2;

}  // namespace

void q4_0_rows_avxvnni(int N, int nb, const std::int8_t* xq, const float* dx,
                       const std::uint8_t* Wq, float* y) noexcept {
    q4_rows<DotAvxVnni, kRowsAvxVnni, kPrefetchAvxVnni>(N, nb, xq, dx, Wq, y);
}

}  // namespace quant
}  // namespace simd_ml
