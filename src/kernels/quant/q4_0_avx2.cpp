/**
 * q4_0_avx2.cpp — the Q4_0 GEMV of the AVX2 table (q4_0.hpp). An AVX2 unit.
 * Without VNNI the 32-byte integer dot is vpmaddubsw (int16 pair sums, at
 * most 3810: q4_0_rows.hpp) then vpmaddwd against ones (int32) and vpaddd
 * into the −8·Σx start: three instructions where vpdpbusd needs one. Where
 * the CPU has AVX-VNNI the table calls the AVX-VNNI unit instead
 * (q4_0_avxvnni.cpp).
 */

#include "q4_0.hpp"
#include "q4_0_rows.hpp"
#include "../gemm/int8_gemm.hpp"

#if !defined(_MSC_VER) && (!defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__))
#  error "q4_0_avx2.cpp is an AVX2 unit: compile with -mavx2 -mfma -mf16c (CMakeLists.txt)"
#endif

namespace simd_ml {
namespace quant {

namespace {

struct DotMaddubs {
    static SIMD_ML_Q4_INLINE __m256i dot_acc(__m256i acc, __m256i u, __m256i s) noexcept {
        return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)));
    }
};

// Rows walked together and prefetch spacing, as the AVX-VNNI path (measured
// there: DESIGN.md §2.9).
constexpr int kRowsAvx2 = 8;
constexpr int kPrefetchAvx2 = 2;

}  // namespace

void q4_0_rows_avx2(int N, int nb, const std::int8_t* xq, const float* dx,
                    const std::uint8_t* Wq, float* y) noexcept {
    q4_rows<DotMaddubs, kRowsAvx2, kPrefetchAvx2>(N, nb, xq, dx, Wq, y);
}

void gemv_q4_0_avx2(int N, int K, const float* x, const std::uint8_t* Wq, float* y) {
#ifdef SIMD_ML_HAVE_AVXVNNI
    if (gemm::avx_vnni_available()) {
        gemv_q8_0_run(N, K, x, Wq, y, &q4_0_rows_avxvnni);
        return;
    }
#endif
    gemv_q8_0_run(N, K, x, Wq, y, &q4_0_rows_avx2);
}

}  // namespace quant
}  // namespace simd_ml
