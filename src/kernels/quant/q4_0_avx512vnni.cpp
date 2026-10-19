/**
 * q4_0_avx512vnni.cpp — the Q4_0 GEMV inner loop with AVX512_VNNI (Cascade
 * Lake, Ice Lake, Sapphire Rapids and later, Zen 4 and later). An
 * AVX-512-VNNI unit; gemv_q4_0_avx512 calls it after CPUID reported
 * AVX512_VNNI. The YMM loop of q4_0_rows.hpp with the EVEX vpdpbusd on
 * 256-bit registers, for the reason given in q8_0_avx512vnni.cpp (an 18-byte
 * block stride; a loop bound by reading memory). Correct under Intel SDE
 * (-icx, -spr); speed not measured (no AVX-512 machine with a stable roof).
 */

#include "q4_0.hpp"
#include "q4_0_rows.hpp"

#if !defined(_MSC_VER) && (!defined(__AVX512F__) || !defined(__AVX512VNNI__) || !defined(__AVX512VL__))
#  error "q4_0_avx512vnni.cpp is an AVX-512-VNNI unit: compile with -mavx512f -mavx512vl -mavx512vnni (CMakeLists.txt)"
#endif

namespace simd_ml {
namespace quant {

namespace {

struct DotAvx512Vnni {
    static SIMD_ML_Q4_INLINE __m256i dot_acc(__m256i acc, __m256i u, __m256i s) noexcept {
        return _mm256_dpbusd_epi32(acc, u, s);
    }
};

constexpr int kRowsAvx512Vnni = 8;
constexpr int kPrefetchAvx512Vnni = 2;

}  // namespace

void q4_0_rows_avx512vnni(int N, int nb, const std::int8_t* xq, const float* dx,
                          const std::uint8_t* Wq, float* y) noexcept {
    q4_rows<DotAvx512Vnni, kRowsAvx512Vnni, kPrefetchAvx512Vnni>(N, nb, xq, dx, Wq, y);
}

}  // namespace quant
}  // namespace simd_ml
