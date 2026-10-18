/**
 * avx512vnni_gemm.cpp — INT8 GEMM (int8_gemm.hpp) with AVX512_VNNI: vpdpbusd
 * on ZMM registers (Cascade Lake, Ice Lake-SP, Sapphire Rapids and later, Zen 4
 * and later). An AVX-512-VNNI unit (AVX-512 flags plus -mavx512vnni); the
 * AVX-512 table's sgemm_u8s8 calls it after CPUID reported AVX512_VNNI.
 *
 * The FP32 AVX-512 kernel's 12×32 tile (24 ZMM accumulators) with K in quads:
 * per quad, 2 B loads and 12 broadcasts feed 24 vpdpbusd of 64 multiply-adds
 * each. Exactness, blocking (KC four times the FP32 KC) and the zero-point
 * correction are as in avxvnni_gemm.cpp. B is packed byte by byte
 * (int8_pack.hpp); a vectorized packing waits until this path can be measured.
 *
 * Correct under Intel SDE (-icx, -spr); speed not measured (no machine here
 * has AVX512_VNNI). Single-threaded.
 */

#include "avx2_gemm_packed.hpp"
#include "int8_gemm.hpp"
#include "int8_pack.hpp"
#include "cache_alloc.hpp"
#include "../../dispatch/cpuid.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#if !defined(_MSC_VER) && (!defined(__AVX512F__) || !defined(__AVX512VNNI__))
#  error "avx512vnni_gemm.cpp is an AVX-512-VNNI unit: compile with -mavx512f -mavx512vnni (CMakeLists.txt)"
#endif
#include <immintrin.h>

namespace simd_ml {
namespace gemm {

namespace {

constexpr int MR = MR512;  // 12
constexpr int NR = NR512;  // 32

#if defined(__GNUC__) || defined(__clang__)
#  define SIMD_ML_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#  define SIMD_ML_ALWAYS_INLINE inline
#endif

/** kernel_12x32_u8s8<R, Tail> — as kernel_6x16_u8s8 in avxvnni_gemm.cpp, on ZMM. */
template <int R, bool Tail>
SIMD_ML_ALWAYS_INLINE void kernel_12x32_u8s8(int kq, const std::uint32_t* __restrict__ Ap,
                                             const std::uint32_t* __restrict__ Bp, int zp,
                                             float a_scale, const float* __restrict__ scales,
                                             float* __restrict__ C, std::ptrdiff_t ldc, int n) noexcept {
    __m512i acc[R][2];
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = _mm512_setzero_si512();
#pragma GCC unroll 2
    for (int q = 0; q < kq; ++q) {
        const __m512i b0 = _mm512_loadu_si512(Bp + q * NR);
        const __m512i b1 = _mm512_loadu_si512(Bp + q * NR + 16);
        const std::uint32_t* a = Ap + q * MR;
        for (int r = 0; r < R; ++r) {
            const __m512i av = _mm512_set1_epi32(static_cast<int>(a[r]));
            acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], av, b0);
            acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], av, b1);
        }
    }
    const __mmask16 m0 = Tail ? static_cast<__mmask16>(n >= 16 ? 0xFFFFu : (1u << n) - 1u) : 0xFFFF;
    const __mmask16 m1 = Tail ? static_cast<__mmask16>(n <= 16 ? 0u : (1u << (n - 16)) - 1u) : 0xFFFF;
    const std::uint32_t* csum = Bp + kq * NR;
    const __m512i zpv = _mm512_set1_epi32(zp);
    const __m512i zc0 = _mm512_mullo_epi32(zpv, _mm512_loadu_si512(csum));
    const __m512i zc1 = _mm512_mullo_epi32(zpv, _mm512_loadu_si512(csum + 16));
    const __m512 as = _mm512_set1_ps(a_scale);
    const __m512 s0 = _mm512_mul_ps(as, _mm512_maskz_loadu_ps(m0, scales));
    const __m512 s1 = _mm512_mul_ps(as, _mm512_maskz_loadu_ps(m1, scales + 16));
    for (int r = 0; r < R; ++r) {
        float* c = C + r * ldc;
        const __m512 v0 = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[r][0], zc0));
        const __m512 v1 = _mm512_cvtepi32_ps(_mm512_sub_epi32(acc[r][1], zc1));
        if (Tail) {
            _mm512_mask_storeu_ps(c, m0, _mm512_fmadd_ps(v0, s0, _mm512_maskz_loadu_ps(m0, c)));
            _mm512_mask_storeu_ps(c + 16, m1, _mm512_fmadd_ps(v1, s1, _mm512_maskz_loadu_ps(m1, c + 16)));
        } else {
            _mm512_storeu_ps(c, _mm512_fmadd_ps(v0, s0, _mm512_loadu_ps(c)));
            _mm512_storeu_ps(c + 16, _mm512_fmadd_ps(v1, s1, _mm512_loadu_ps(c + 16)));
        }
    }
}

using KernelFn = void (*)(int, const std::uint32_t*, const std::uint32_t*, int, float, const float*,
                          float*, std::ptrdiff_t, int);

template <bool Tail, std::size_t... I>
constexpr auto make_row_table(std::index_sequence<I...>) {
    // Entry r runs a kernel for r + 1 rows.
    return std::array<KernelFn, sizeof...(I)>{{&kernel_12x32_u8s8<int(I) + 1, Tail>...}};
}

/** pack_b — op(B) into 32-column micro-panels of k-quads (byte by byte) plus column sums. */
void pack_b(const std::int8_t* B, std::ptrdiff_t ldb, bool trans, int kc, int nc, std::uint32_t* Bp) noexcept {
    const int kq = kquads(kc);
    const __m512i ones = _mm512_set1_epi8(1);
    for (int j = 0; j < nc; j += NR) {
        const int cols = std::min(NR, nc - j);
        std::uint32_t* dst = Bp + static_cast<std::ptrdiff_t>(j / NR) * b_panel_words<NR>(kc);
        pack_b_s8_panel<NR>(trans ? B + j * ldb : B + j, ldb, trans, kc, cols, 0, dst);
        __m512i s0 = _mm512_setzero_si512(), s1 = _mm512_setzero_si512();
        for (int p = 0; p < kq; ++p) {
            s0 = _mm512_dpbusd_epi32(s0, ones, _mm512_loadu_si512(dst + p * NR));
            s1 = _mm512_dpbusd_epi32(s1, ones, _mm512_loadu_si512(dst + p * NR + 16));
        }
        _mm512_storeu_si512(dst + kq * NR, s0);
        _mm512_storeu_si512(dst + kq * NR + 16, s1);
    }
}

struct Blocking { int kc, mc, nc; };

const Blocking& blocking() noexcept {
    static const Blocking b = []() noexcept {
        const dispatch::GemmBlocking g = dispatch::gemm_blocking(MR, NR);
        return Blocking{4 * g.kc, g.mc, g.nc};
    }();
    return b;
}

struct PackBuffers {
    AlignedUniquePtr<std::uint32_t> a, b;
    std::size_t a_cap = 0, b_cap = 0;
    std::uint32_t* get_a(std::size_t n) {
        if (n > a_cap) { a = make_aligned_array<std::uint32_t>(n); a_cap = n; }
        return a.get();
    }
    std::uint32_t* get_b(std::size_t n) {
        if (n > b_cap) { b = make_aligned_array<std::uint32_t>(n); b_cap = n; }
        return b.get();
    }
};

PackBuffers& thread_buffers() {
    thread_local PackBuffers buf;
    return buf;
}

}  // namespace

void sgemm_u8s8_avx512vnni(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                           int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                           const float* b_scales, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    scale_matrix_c(C, M, N, ldc, 0.0f);
    if (K == 0) return;
    static constexpr auto full = make_row_table<false>(std::make_index_sequence<MR>{});
    static constexpr auto tail = make_row_table<true>(std::make_index_sequence<MR>{});
    const Blocking& bl = blocking();
    PackBuffers& buf = thread_buffers();
    const int kc_max = std::min(K, bl.kc), mc_max = std::min(M, bl.mc);
    for (int jc = 0; jc < N; jc += bl.nc) {
        const int nc = std::min(N - jc, bl.nc);
        std::uint32_t* Bp = buf.get_b(static_cast<std::size_t>((nc + NR - 1) / NR) * b_panel_words<NR>(kc_max));
        std::uint32_t* Ap = buf.get_a(static_cast<std::size_t>((mc_max + MR - 1) / MR) * MR * kquads(kc_max));
        for (int pc = 0; pc < K; pc += bl.kc) {
            const int kc = std::min(K - pc, bl.kc), kq = kquads(kc);
            pack_b(trans_b ? B + static_cast<std::ptrdiff_t>(jc) * ldb + pc : B + static_cast<std::ptrdiff_t>(pc) * ldb + jc,
                   ldb, trans_b, kc, nc, Bp);
            for (int ic = 0; ic < M; ic += bl.mc) {
                const int mc = std::min(M - ic, bl.mc);
                pack_a_u8<MR>(A + static_cast<std::ptrdiff_t>(ic) * lda + pc, lda, mc, kc, Ap);
                for (int jr = 0; jr < nc; jr += NR) {
                    const int n = std::min(NR, nc - jr);
                    const std::uint32_t* Bm = Bp + static_cast<std::ptrdiff_t>(jr / NR) * b_panel_words<NR>(kc);
                    const float* sc = b_scales + jc + jr;
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int rows = std::min(MR, mc - ir);
                        const std::uint32_t* Am = Ap + static_cast<std::ptrdiff_t>(ir / MR) * MR * kq;
                        float* Cm = C + static_cast<std::ptrdiff_t>(ic + ir) * ldc + jc + jr;
                        if (rows == MR && n == NR)
                            kernel_12x32_u8s8<MR, false>(kq, Am, Bm, a_zero_point, a_scale, sc, Cm, ldc, n);
                        else
                            (n == NR ? full : tail)[rows - 1](kq, Am, Bm, a_zero_point, a_scale, sc, Cm, ldc, n);
                    }
                }
            }
        }
    }
}

}  // namespace gemm
}  // namespace simd_ml
