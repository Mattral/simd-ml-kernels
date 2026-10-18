/**
 * avxvnni_gemm.cpp — INT8 GEMM (uint8 × int8, per-column scales, FP32 output;
 * int8_gemm.hpp) with AVX-VNNI: VEX-encoded vpdpbusd on YMM registers (Alder
 * Lake, Lunar Lake, Sapphire Rapids and later). An AVX-VNNI unit (AVX2 flags
 * plus -mavxvnni); the AVX2 table's sgemm_u8s8 calls it after CPUID reported
 * AVX-VNNI.
 *
 * ─── Why this is faster than FP32 ────────────────────────────────────────────
 * One vpdpbusd ymm does 8 lanes × 4 byte products and adds them to int32
 * lanes: 32 multiply-adds, against 8 for one vfmadd231ps ymm, and it issues
 * on the same two vector ports. The tile is the FP32 kernel's 6×16 (12 YMM
 * accumulators + 2 B vectors + 1 broadcast = 15 of 16 registers), with K in
 * quads: per quad 2 B loads and 6 four-byte A broadcasts feed 12 vpdpbusd.
 *
 * ─── Exactness and blocking ──────────────────────────────────────────────────
 * Within one KC block the int32 sums are exact: |Σ a·b| <= KC·255·128 < 2^31
 * for KC up to 65793. The block's sum minus zp·(column sum of B) is converted
 * to FP32, scaled by a_scale·b_scales[j] and added to C, so the only rounding
 * is per block. KC is four times the FP32 KC (one byte per element instead of
 * four, so the KC×16 B micro-panel takes the same half of L1D); MC and NC are
 * the FP32 ones (the A block and the B panel then take a quarter of the bytes).
 *
 * Single-threaded (OpenMP builds included): the INT8 GEMM has no threaded path.
 */

#include "avx2_gemm_packed.hpp"
#include "int8_gemm.hpp"
#include "int8_pack.hpp"
#include "cache_alloc.hpp"
#include "../../dispatch/cpuid.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if !defined(_MSC_VER) && (!defined(__AVX2__) || !defined(__AVXVNNI__))
#  error "avxvnni_gemm.cpp is an AVX-VNNI unit: compile with -mavx2 -mfma -mavxvnni (CMakeLists.txt)"
#endif
#include <immintrin.h>

namespace simd_ml {
namespace gemm {

namespace {

constexpr int MR = MR_AVX2;  // 6
constexpr int NR = NR_AVX2;  // 16

#if defined(__GNUC__) || defined(__clang__)
#  define SIMD_ML_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#  define SIMD_ML_ALWAYS_INLINE inline
#endif

/**
 * kernel_6x16_u8s8<R, Tail> — C[0..R)[0..n) += s_j · (Σ_q Ap·Bp − zp·csum_j)
 * over kq k-quads, s_j = a_scale·b_scales[j]. Tail masks the C and scale
 * accesses past column n (packed B is zero there).
 */
template <int R, bool Tail>
SIMD_ML_ALWAYS_INLINE void kernel_6x16_u8s8(int kq, const std::uint32_t* __restrict__ Ap,
                                            const std::uint32_t* __restrict__ Bp, int zp,
                                            float a_scale, const float* __restrict__ scales,
                                            float* __restrict__ C, std::ptrdiff_t ldc, int n) noexcept {
    __m256i acc[R][2];
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_si256();
#pragma GCC unroll 4
    for (int q = 0; q < kq; ++q) {
        const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + q * NR));
        const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Bp + q * NR + 8));
        const std::uint32_t* a = Ap + q * MR;
        for (int r = 0; r < R; ++r) {
            const __m256i av = _mm256_set1_epi32(static_cast<int>(a[r]));
            acc[r][0] = _mm256_dpbusd_avx_epi32(acc[r][0], av, b0);
            acc[r][1] = _mm256_dpbusd_avx_epi32(acc[r][1], av, b1);
        }
    }
    __m256i m0 = _mm256_set1_epi32(-1), m1 = m0;
    if (Tail) {
        const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        m0 = _mm256_cmpgt_epi32(_mm256_set1_epi32(n), lane);
        m1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(n - 8), lane);
    }
    const std::uint32_t* csum = Bp + kq * NR;
    const __m256i zpv = _mm256_set1_epi32(zp);
    const __m256i zc0 = _mm256_mullo_epi32(zpv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(csum)));
    const __m256i zc1 = _mm256_mullo_epi32(zpv, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(csum + 8)));
    const __m256 as = _mm256_set1_ps(a_scale);
    const __m256 s0 = _mm256_mul_ps(as, Tail ? _mm256_maskload_ps(scales, m0) : _mm256_loadu_ps(scales));
    const __m256 s1 = _mm256_mul_ps(as, Tail ? _mm256_maskload_ps(scales + 8, m1) : _mm256_loadu_ps(scales + 8));
    for (int r = 0; r < R; ++r) {
        float* c = C + r * ldc;
        const __m256 v0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][0], zc0));
        const __m256 v1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(acc[r][1], zc1));
        if (Tail) {
            _mm256_maskstore_ps(c, m0, _mm256_fmadd_ps(v0, s0, _mm256_maskload_ps(c, m0)));
            _mm256_maskstore_ps(c + 8, m1, _mm256_fmadd_ps(v1, s1, _mm256_maskload_ps(c + 8, m1)));
        } else {
            _mm256_storeu_ps(c, _mm256_fmadd_ps(v0, s0, _mm256_loadu_ps(c)));
            _mm256_storeu_ps(c + 8, _mm256_fmadd_ps(v1, s1, _mm256_loadu_ps(c + 8)));
        }
    }
}

template <bool Tail>
void run_rows(int rows, int kq, const std::uint32_t* Ap, const std::uint32_t* Bp, int zp, float a_scale,
              const float* scales, float* C, std::ptrdiff_t ldc, int n) noexcept {
    switch (rows) {
        case 6: kernel_6x16_u8s8<6, Tail>(kq, Ap, Bp, zp, a_scale, scales, C, ldc, n); break;
        case 5: kernel_6x16_u8s8<5, Tail>(kq, Ap, Bp, zp, a_scale, scales, C, ldc, n); break;
        case 4: kernel_6x16_u8s8<4, Tail>(kq, Ap, Bp, zp, a_scale, scales, C, ldc, n); break;
        case 3: kernel_6x16_u8s8<3, Tail>(kq, Ap, Bp, zp, a_scale, scales, C, ldc, n); break;
        case 2: kernel_6x16_u8s8<2, Tail>(kq, Ap, Bp, zp, a_scale, scales, C, ldc, n); break;
        case 1: kernel_6x16_u8s8<1, Tail>(kq, Ap, Bp, zp, a_scale, scales, C, ldc, n); break;
        default: break;
    }
}

/** transpose8x8 — r[q] becomes column q of the 8×8 block of 32-bit words in r[0..7]. */
inline void transpose8x8(__m256 r[8]) noexcept {
    const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
    const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
    const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
    const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
    const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    r[0] = _mm256_permute2f128_ps(u0, u4, 0x20); r[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
    r[2] = _mm256_permute2f128_ps(u2, u6, 0x20); r[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
    r[4] = _mm256_permute2f128_ps(u0, u4, 0x31); r[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
    r[6] = _mm256_permute2f128_ps(u2, u6, 0x31); r[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}

/**
 * pack_b — op(B)[0..kc)[0..nc) into 16-column micro-panels of k-quads plus
 * column sums (int8_pack.hpp). Full panels, B stored K×N: four rows of 16
 * bytes interleaved with vpunpck{l,h}bw / vpunpck{l,h}wd into 4-byte columns.
 * Full panels, trans (B stored N×K, each column contiguous along k): 8×8
 * transposes of 32-bit words (8 columns × 8 quads). The rest byte by byte.
 * Column sums: vpdpbusd of the packed quads with all-ones bytes.
 */
void pack_b(const std::int8_t* B, std::ptrdiff_t ldb, bool trans, int kc, int nc, std::uint32_t* Bp) noexcept {
    const int kq = kquads(kc), full = kc / 4;
    const __m256i ones = _mm256_set1_epi8(1);
    for (int j = 0; j < nc; j += NR) {
        const int cols = std::min(NR, nc - j);
        std::uint32_t* dst = Bp + static_cast<std::ptrdiff_t>(j / NR) * b_panel_words<NR>(kc);
        int q = 0;
        if (cols == NR && !trans) {
            for (; q < full; ++q) {
                const std::int8_t* src = B + static_cast<std::ptrdiff_t>(4 * q) * ldb + j;
                const __m128i r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
                const __m128i r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + ldb));
                const __m128i r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 2 * ldb));
                const __m128i r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 3 * ldb));
                const __m128i t0 = _mm_unpacklo_epi8(r0, r1), t1 = _mm_unpackhi_epi8(r0, r1);
                const __m128i t2 = _mm_unpacklo_epi8(r2, r3), t3 = _mm_unpackhi_epi8(r2, r3);
                __m128i* d = reinterpret_cast<__m128i*>(dst + q * NR);
                _mm_storeu_si128(d, _mm_unpacklo_epi16(t0, t2));
                _mm_storeu_si128(d + 1, _mm_unpackhi_epi16(t0, t2));
                _mm_storeu_si128(d + 2, _mm_unpacklo_epi16(t1, t3));
                _mm_storeu_si128(d + 3, _mm_unpackhi_epi16(t1, t3));
            }
        } else if (cols == NR && trans) {
            for (; q + 8 <= full; q += 8)
                for (int h = 0; h < NR; h += 8) {
                    __m256 r[8];
                    for (int c = 0; c < 8; ++c)
                        r[c] = _mm256_castsi256_ps(_mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(B + (j + h + c) * ldb + 4 * q)));
                    transpose8x8(r);
                    for (int t = 0; t < 8; ++t)
                        _mm256_storeu_ps(reinterpret_cast<float*>(dst + (q + t) * NR + h), r[t]);
                }
        }
        pack_b_s8_panel<NR>(trans ? B + j * ldb : B + j, ldb, trans, kc, cols, q, dst);
        __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
        for (int p = 0; p < kq; ++p) {
            s0 = _mm256_dpbusd_avx_epi32(s0, ones, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + p * NR)));
            s1 = _mm256_dpbusd_avx_epi32(s1, ones, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + p * NR + 8)));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + kq * NR), s0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + kq * NR + 8), s1);
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

void sgemm_u8s8_avxvnni(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                        int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                        const float* b_scales, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    scale_matrix_c(C, M, N, ldc, 0.0f);
    if (K == 0) return;
    const Blocking& bl = blocking();
    PackBuffers& buf = thread_buffers();
    const int kc_max = std::min(K, bl.kc), mc_max = std::min(M, bl.mc);
    for (int jc = 0; jc < N; jc += bl.nc) {                       // loop 1
        const int nc = std::min(N - jc, bl.nc);
        std::uint32_t* Bp = buf.get_b(static_cast<std::size_t>((nc + NR - 1) / NR) * b_panel_words<NR>(kc_max));
        std::uint32_t* Ap = buf.get_a(static_cast<std::size_t>((mc_max + MR - 1) / MR) * MR * kquads(kc_max));
        for (int pc = 0; pc < K; pc += bl.kc) {                   // loop 2
            const int kc = std::min(K - pc, bl.kc), kq = kquads(kc);
            pack_b(trans_b ? B + static_cast<std::ptrdiff_t>(jc) * ldb + pc : B + static_cast<std::ptrdiff_t>(pc) * ldb + jc,
                   ldb, trans_b, kc, nc, Bp);
            for (int ic = 0; ic < M; ic += bl.mc) {               // loop 3
                const int mc = std::min(M - ic, bl.mc);
                pack_a_u8<MR>(A + static_cast<std::ptrdiff_t>(ic) * lda + pc, lda, mc, kc, Ap);
                for (int jr = 0; jr < nc; jr += NR) {             // loop 4
                    const int n = std::min(NR, nc - jr);
                    const std::uint32_t* Bm = Bp + static_cast<std::ptrdiff_t>(jr / NR) * b_panel_words<NR>(kc);
                    const float* sc = b_scales + jc + jr;
                    for (int ir = 0; ir < mc; ir += MR) {         // loop 5
                        const int rows = std::min(MR, mc - ir);
                        const std::uint32_t* Am = Ap + static_cast<std::ptrdiff_t>(ir / MR) * MR * kq;
                        float* Cm = C + static_cast<std::ptrdiff_t>(ic + ir) * ldc + jc + jr;
                        if (rows == MR && n == NR)
                            kernel_6x16_u8s8<MR, false>(kq, Am, Bm, a_zero_point, a_scale, sc, Cm, ldc, n);
                        else if (n == NR)
                            run_rows<false>(rows, kq, Am, Bm, a_zero_point, a_scale, sc, Cm, ldc, n);
                        else
                            run_rows<true>(rows, kq, Am, Bm, a_zero_point, a_scale, sc, Cm, ldc, n);
                    }
                }
            }
        }
    }
}

}  // namespace gemm
}  // namespace simd_ml
