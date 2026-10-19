/**
 * q8_0_avx512vnni.cpp — the Q8_0 GEMV inner loop with AVX512_VNNI (Cascade
 * Lake, Ice Lake, Sapphire Rapids and later, Zen 4 and later). An
 * AVX-512-VNNI unit; gemv_q8_0_avx512 calls it after CPUID reported
 * AVX512_VNNI.
 *
 * It runs the YMM loop of q8_0_rows.hpp with the EVEX vpdpbusd on 256-bit
 * registers (AVX512_VNNI + AVX512VL), not ZMM: a block is 32 quants at a
 * 34-byte stride, so a ZMM register would need two loads and a lane insert
 * per two blocks, plus a per-lane scale vector, for a loop bound by reading
 * memory, not by its instruction count. CPUs with AVX512_VNNI but no AVX-VNNI
 * (Cascade Lake, Ice Lake, Zen 4) get one dot instruction per 32 weights
 * instead of the AVX2 pair. Correct under Intel SDE (-icx, -spr); speed not
 * measured (no AVX-512 machine with a stable roof here).
 */

#include "q8_0.hpp"
#include "q8_0_rows.hpp"

#if !defined(_MSC_VER) && (!defined(__AVX512F__) || !defined(__AVX512VNNI__) || !defined(__AVX512VL__))
#  error "q8_0_avx512vnni.cpp is an AVX-512-VNNI unit: compile with -mavx512f -mavx512vl -mavx512vnni (CMakeLists.txt)"
#endif

namespace simd_ml {
namespace quant {

namespace {

struct DotAvx512Vnni {
    static SIMD_ML_Q8_INLINE __m256i dot(__m256i u, __m256i s) noexcept {
        return _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
    }
};

constexpr int kRowsAvx512Vnni = 8;

}  // namespace

void q8_0_rows_avx512vnni(int N, int nb, const std::int8_t* xq, const float* dx,
                          const std::uint8_t* Wq, float* y) noexcept {
    q8_rows<DotAvx512Vnni, kRowsAvx512Vnni>(N, nb, xq, dx, Wq, y);
}

}  // namespace quant
}  // namespace simd_ml
