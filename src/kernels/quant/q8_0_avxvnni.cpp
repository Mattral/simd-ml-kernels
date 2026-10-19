/**
 * q8_0_avxvnni.cpp — the Q8_0 GEMV inner loop with AVX-VNNI (VEX vpdpbusd on
 * YMM: Alder Lake, Lunar Lake, Sapphire Rapids and later). An AVX-VNNI unit;
 * gemv_q8_0_avx2 calls it after CPUID reported AVX-VNNI. One vpdpbusd per
 * 32 weights replaces vpmaddubsw + vpmaddwd (q8_0_rows.hpp for the rest).
 */

#include "q8_0.hpp"
#include "q8_0_rows.hpp"

#if !defined(_MSC_VER) && (!defined(__AVX2__) || !defined(__AVXVNNI__))
#  error "q8_0_avxvnni.cpp is an AVX-VNNI unit: compile with -mavx2 -mfma -mavxvnni (CMakeLists.txt)"
#endif

namespace simd_ml {
namespace quant {

namespace {

struct DotAvxVnni {
    static SIMD_ML_Q8_INLINE __m256i dot(__m256i u, __m256i s) noexcept {
        return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s);
    }
};

// Rows walked together. Dev runs on the reference laptop, % of the read roof
// (no software prefetch): 1 row 50-63%, 2 rows 55-63%, 4 rows 60-76%, 8 rows
// 70-91%. 8 accumulators + x, |x| and dx fit the 16 YMM registers.
constexpr int kRowsAvxVnni = 8;

}  // namespace

void q8_0_rows_avxvnni(int N, int nb, const std::int8_t* xq, const float* dx,
                       const std::uint8_t* Wq, float* y) noexcept {
    q8_rows<DotAvxVnni, kRowsAvxVnni>(N, nb, xq, dx, Wq, y);
}

}  // namespace quant
}  // namespace simd_ml
