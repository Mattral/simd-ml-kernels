/**
 * avx512bf16_gemm.cpp — BF16-input GEMM with vdpbf16ps (AVX512_BF16). An
 * AVX-512-BF16 unit (AVX-512 flags plus -mavx512bf16); the AVX-512 table's
 * sgemm_bf16 calls it only after CPUID reported AVX512_BF16 (Cooper Lake,
 * Sapphire Rapids and later, Zen 4 and later).
 *
 * vdpbf16ps zmm_acc, zmm_a, zmm_b: for each of the 16 FP32 lanes,
 * acc += a[2i+1]·b[2i+1], then acc += a[2i]·b[2i] (Intel SDM): two BF16
 * products per lane per instruction, against one FP32 product per lane for
 * vfmadd231ps. Products of two BF16 values are exact in FP32, and each add is
 * rounded to nearest even, so the error bound is the FP32 GEMM's. The
 * instruction ignores MXCSR: BF16 subnormal inputs are read as zero (DAZ) and
 * subnormal results are flushed to zero (FTZ), unlike the conversion path.
 *
 * Layout: the 12×32 micro-tile of the FP32 AVX-512 kernel (24 accumulators),
 * with K taken in pairs. Packed A holds, per k-pair and row, one 32-bit word
 * (A(r, 2q) in the low half, A(r, 2q+1) in the high half), broadcast to every
 * lane; packed B holds, per k-pair, 32 columns × 2 BF16 = two ZMM vectors, the
 * pair of column c in dword c. Odd K is padded with a zero BF16.
 *
 * Correctness is checked under Intel SDE (-spr) and on hardware with the
 * instruction; its speed has not been measured (no machine available here
 * has AVX512_BF16; docs/BENCHMARKS.md). Single-threaded: with more than one
 * OpenMP thread the AVX-512 table uses the threaded conversion path instead.
 */

#include "avx2_gemm_packed.hpp"
#include "bf16_gemm.hpp"
#include "cache_alloc.hpp"
#include "../../dispatch/cpuid.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if !defined(_MSC_VER) && (!defined(__AVX512F__) || !defined(__AVX512BF16__))
#  error "avx512bf16_gemm.cpp is an AVX-512-BF16 unit: compile with -mavx512f -mavx512bf16 (CMakeLists.txt)"
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

/** kernel_12x32_bf16<R, Tail> — C[0..R)[0..n) += alpha · Ap · Bp over kp k-pairs. */
template <int R, bool Tail>
SIMD_ML_ALWAYS_INLINE void kernel_12x32_bf16(int kp, const std::uint32_t* __restrict__ Ap,
                                             const std::uint16_t* __restrict__ Bp,
                                             float* __restrict__ C, std::ptrdiff_t ldc, int n,
                                             float alpha) noexcept {
    __m512 acc[R][2];
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = _mm512_setzero_ps();
#pragma GCC unroll 2
    for (int q = 0; q < kp; ++q) {
        const __m512bh b0 = (__m512bh)_mm512_loadu_si512(Bp + q * 2 * NR);
        const __m512bh b1 = (__m512bh)_mm512_loadu_si512(Bp + q * 2 * NR + NR);
        const std::uint32_t* a = Ap + q * MR;
        for (int r = 0; r < R; ++r) {
            const __m512bh av = (__m512bh)_mm512_set1_epi32(static_cast<int>(a[r]));
            acc[r][0] = _mm512_dpbf16_ps(acc[r][0], av, b0);
            acc[r][1] = _mm512_dpbf16_ps(acc[r][1], av, b1);
        }
    }
    const __m512 alpha_v = _mm512_set1_ps(alpha);
    const __mmask16 m0 = Tail ? static_cast<__mmask16>(n >= 16 ? 0xFFFFu : (1u << n) - 1u) : 0xFFFF;
    const __mmask16 m1 = Tail ? static_cast<__mmask16>(n <= 16 ? 0u : (1u << (n - 16)) - 1u) : 0xFFFF;
    for (int r = 0; r < R; ++r) {
        float* c = C + r * ldc;
        if (Tail) {
            _mm512_mask_storeu_ps(c, m0, _mm512_fmadd_ps(acc[r][0], alpha_v, _mm512_maskz_loadu_ps(m0, c)));
            _mm512_mask_storeu_ps(c + 16, m1, _mm512_fmadd_ps(acc[r][1], alpha_v, _mm512_maskz_loadu_ps(m1, c + 16)));
        } else {
            _mm512_storeu_ps(c, _mm512_fmadd_ps(acc[r][0], alpha_v, _mm512_loadu_ps(c)));
            _mm512_storeu_ps(c + 16, _mm512_fmadd_ps(acc[r][1], alpha_v, _mm512_loadu_ps(c + 16)));
        }
    }
}

template <bool Tail>
void run_rows(int rows, int kp, const std::uint32_t* Ap, const std::uint16_t* Bp, float* C,
              std::ptrdiff_t ldc, int n, float alpha) noexcept {
    switch (rows) {
        case 12: kernel_12x32_bf16<12, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        case 11: kernel_12x32_bf16<11, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        case 10: kernel_12x32_bf16<10, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        case 9:  kernel_12x32_bf16<9, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        case 8:  kernel_12x32_bf16<8, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        case 7:  kernel_12x32_bf16<7, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        case 6:  kernel_12x32_bf16<6, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        case 5:  kernel_12x32_bf16<5, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        case 4:  kernel_12x32_bf16<4, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        case 3:  kernel_12x32_bf16<3, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        case 2:  kernel_12x32_bf16<2, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        case 1:  kernel_12x32_bf16<1, Tail>(kp, Ap, Bp, C, ldc, n, alpha); break;
        default: break;
    }
}

/** pack_a — A[0..mc)[0..kc) into 12-row micro-panels of k-pairs; rows past mc and k = kc (odd) zero. */
void pack_a(const std::uint16_t* A, std::ptrdiff_t lda, int mc, int kc, std::uint32_t* Ap) noexcept {
    const int kp = (kc + 1) / 2;
    for (int i = 0; i < mc; i += MR) {
        const int rows = std::min(MR, mc - i);
        std::uint32_t* dst = Ap + static_cast<std::ptrdiff_t>(i / MR) * MR * kp;
        for (int q = 0; q < kp; ++q)
            for (int r = 0; r < MR; ++r) {
                std::uint32_t w = 0;
                if (r < rows) {
                    const std::uint16_t* src = A + (i + r) * lda + 2 * q;
                    w = src[0] | (2 * q + 1 < kc ? static_cast<std::uint32_t>(src[1]) << 16 : 0u);
                }
                dst[q * MR + r] = w;
            }
    }
}

/**
 * pack_b — B[0..kc)[0..nc) into 32-column panels of k-pairs:
 * Bp[(j/32)·kp·64 + q·64 + 2c + h] = B(2q + h, j + c); zero past nc and kc.
 * Full panels interleave two B rows with vpunpck{l,h}wd (per 128-bit lane:
 * columns 0-3 and 4-7 of each 8) and restore column order with vpermt2q.
 */
void pack_b(const std::uint16_t* B, std::ptrdiff_t ldb, int kc, int nc, std::uint16_t* Bp) noexcept {
    const int kp = (kc + 1) / 2;
    const __m512i idx0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i idx1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    for (int j = 0; j < nc; j += NR) {
        const int cols = std::min(NR, nc - j);
        std::uint16_t* dst = Bp + static_cast<std::ptrdiff_t>(j / NR) * kp * 2 * NR;
        int q = 0;
        if (cols == NR) {
            for (; 2 * q + 1 < kc; ++q) {
                const __m512i r0 = _mm512_loadu_si512(B + (2 * q) * ldb + j);
                const __m512i r1 = _mm512_loadu_si512(B + (2 * q + 1) * ldb + j);
                const __m512i lo = _mm512_unpacklo_epi16(r0, r1), hi = _mm512_unpackhi_epi16(r0, r1);
                _mm512_storeu_si512(dst + q * 2 * NR, _mm512_permutex2var_epi64(lo, idx0, hi));
                _mm512_storeu_si512(dst + q * 2 * NR + NR, _mm512_permutex2var_epi64(lo, idx1, hi));
            }
        }
        for (; q < kp; ++q)   // odd-K pair, and partial panels
            for (int c = 0; c < NR; ++c)
                for (int h = 0; h < 2; ++h) {
                    const int p = 2 * q + h;
                    dst[q * 2 * NR + 2 * c + h] = (c < cols && p < kc) ? B[p * ldb + j + c] : 0;
                }
    }
}

struct Blocking { int kc, mc, nc; };

const Blocking& blocking() noexcept {
    // The FP32 12×32 blocking: KC (a multiple of 8, so even) counts k, not
    // k-pairs; the BF16 panels are half the bytes of the FP32 ones.
    static const Blocking b = []() noexcept {
        const dispatch::GemmBlocking g = dispatch::gemm_blocking(MR, NR);
        return Blocking{g.kc, g.mc, g.nc};
    }();
    return b;
}

struct PackBuffers {
    AlignedUniquePtr<std::uint32_t> a;
    AlignedUniquePtr<std::uint16_t> b;
    std::size_t a_cap = 0, b_cap = 0;
    std::uint32_t* get_a(std::size_t n) {
        if (n > a_cap) { a = make_aligned_array<std::uint32_t>(n); a_cap = n; }
        return a.get();
    }
    std::uint16_t* get_b(std::size_t n) {
        if (n > b_cap) { b = make_aligned_array<std::uint16_t>(n); b_cap = n; }
        return b.get();
    }
};

PackBuffers& thread_buffers() {
    thread_local PackBuffers buf;
    return buf;
}

}  // namespace

void sgemm_bf16_avx512bf16(int M, int N, int K, float alpha, const std::uint16_t* A, int lda,
                           const std::uint16_t* B, int ldb, float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    scale_matrix_c(C, M, N, ldc, beta);
    if (K == 0) return;
    const Blocking& bl = blocking();
    PackBuffers& buf = thread_buffers();
    const int kp_max = (std::min(K, bl.kc) + 1) / 2;
    const int mc_max = std::min(M, bl.mc);
    for (int jc = 0; jc < N; jc += bl.nc) {                       // loop 1
        const int nc = std::min(N - jc, bl.nc);
        std::uint16_t* Bp = buf.get_b(static_cast<std::size_t>(kp_max) * 2 * ((nc + NR - 1) / NR * NR));
        std::uint32_t* Ap = buf.get_a(static_cast<std::size_t>(kp_max) * ((mc_max + MR - 1) / MR * MR));
        for (int pc = 0; pc < K; pc += bl.kc) {                   // loop 2
            const int kc = std::min(K - pc, bl.kc), kp = (kc + 1) / 2;
            pack_b(B + static_cast<std::ptrdiff_t>(pc) * ldb + jc, ldb, kc, nc, Bp);
            for (int ic = 0; ic < M; ic += bl.mc) {               // loop 3
                const int mc = std::min(M - ic, bl.mc);
                pack_a(A + static_cast<std::ptrdiff_t>(ic) * lda + pc, lda, mc, kc, Ap);
                for (int jr = 0; jr < nc; jr += NR) {             // loop 4
                    const int n = std::min(NR, nc - jr);
                    const std::uint16_t* Bm = Bp + static_cast<std::ptrdiff_t>(jr / NR) * kp * 2 * NR;
                    for (int ir = 0; ir < mc; ir += MR) {         // loop 5
                        const int rows = std::min(MR, mc - ir);
                        const std::uint32_t* Am = Ap + static_cast<std::ptrdiff_t>(ir / MR) * MR * kp;
                        float* Cm = C + static_cast<std::ptrdiff_t>(ic + ir) * ldc + jc + jr;
                        if (rows == MR && n == NR) kernel_12x32_bf16<MR, false>(kp, Am, Bm, Cm, ldc, n, alpha);
                        else if (n == NR) run_rows<false>(rows, kp, Am, Bm, Cm, ldc, n, alpha);
                        else run_rows<true>(rows, kp, Am, Bm, Cm, ldc, n, alpha);
                    }
                }
            }
        }
    }
}

}  // namespace gemm
}  // namespace simd_ml
