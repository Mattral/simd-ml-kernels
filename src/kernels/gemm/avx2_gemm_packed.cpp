/**
 * avx2_gemm_packed.cpp — FP32 GEMM, AVX2 + FMA: one 6×16 micro-kernel, used
 * directly on A and B for small products and inside the Goto/BLIS five loops
 * with packed panels otherwise. docs/DESIGN.md §2 has the full story; the
 * numbers that decided each choice are in the comments below.
 *
 * ─── Why 6×16 (12 YMM accumulators) ──────────────────────────────────────────
 * Ports and latency: Haswell to Lion Cove issue 2 FMAs per cycle with a
 * latency of 4 cycles (Lion Cove, Golden Cove, Skylake; 5 on Haswell), so at
 * least 2 × 4 = 8 independent accumulators must be in flight to keep both FMA
 * pipes busy. The old 8×8 kernel had exactly 8 (no slack for a late load);
 * 6×16 has 12.
 * Registers: 16 YMM. 12 accumulators + 2 B vectors + 1 broadcast A = 15.
 * (7×16 or 8×16 would need 16+ accumulators alone.)
 * Loads per FMA: per k-step 2 B loads + 6 A broadcasts feed 12 FMAs, 0.67
 * loads per FMA, against 9 loads per 8 FMAs (1.13) for 8×8 and 7 per 12
 * (0.58) for 4×24. Two or three load ports sustain 6×16 at full FMA rate.
 * Measured (Lunar Lake P-core, one generic packed driver, persistent buffers,
 * square N, OpenBLAS 0.3.34 = 100%): 8×8 84-91%, 6×16 98-105%, 4×24 95-103%
 * for N = 256..2048; 6×16 was best or tied at every size from 128 up.
 *
 * ─── Block sizes, from CPUID leaf 4 (dispatch::gemm_blocking) ───────────────
 * KC: one KC×16 B micro-panel stays in L1 while it is reused for every 6-row
 *     A micro-panel of the MC block, so it gets half of L1D (the other half
 *     holds the streaming A micro-panel and the C tile): KC = L1D/2/(16·4).
 *     48 KiB (Lion Cove) -> 384; 32 KiB (Skylake, Coffee Lake) -> 256.
 * MC: the packed MC×KC A block stays in L2 across the jr loop. A quarter of
 *     L2 for it: on Lunar Lake (2.5 MiB L2) MC = 426; a sweep at N = 2048 put
 *     MC = 852 (half of L2) 5-10% below MC <= 576 at every KC, and anything
 *     from 72 to 576 within noise (the B panel is re-read from L3 only once
 *     per MC rows, 2/MC bytes per flop).
 * NC: the packed KC×NC B panel lives in the last-level cache: half of it,
 *     LNL 8 MiB L3 -> NC = 2720.
 * On hybrid CPUs leaf 4 describes the core the first GEMM ran on.
 *
 * ─── Small products: the direct path ─────────────────────────────────────────
 * Packing copies all of A and B once per call; for small products that is a
 * large share of the work, so sgemm_avx2_prefers_direct sends them to the
 * same kernel reading A (6 row streams) and B (rows of 16 floats) in place.
 * Before this change, max(M,N,K) <= 128 used a one-accumulator loop, bound by
 * FMA latency (one FMA per 4 cycles: 24% of OpenBLAS at N = 128), and every
 * packed call allocated and freed 2.1 MiB of buffers (34 µs per call measured,
 * plus first-touch page faults): 43% at N = 128 forced through the packed path,
 * 82% with the buffers kept. Buffers are now per thread, grown on demand and
 * kept, so a call allocates nothing in the steady state.
 */

#include "avx2_gemm_packed.hpp"
#include "bf16_gemm.hpp"
#include "f16_gemm.hpp"
#include "int8_gemm.hpp"
#include "int8_rows.hpp"
#include "cache_alloc.hpp"
#include "../../dispatch/cpuid.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__)
#  error "avx2_gemm_packed.cpp is an AVX2 unit: compile with -mavx2 -mfma -mf16c (CMakeLists.txt)"
#endif
#include <immintrin.h>
#ifdef SIMD_ML_OPENMP
#  include <omp.h>
#endif

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

// ─── Input element types ─────────────────────────────────────────────────────
// The packing routines read A and B through one of these and write FP32, so
// FP16 and BF16 inputs are converted once per packed element and the
// micro-kernel only ever sees floats. F32 compiles to the plain loads.
struct F32 {
    using T = float;
    static SIMD_ML_ALWAYS_INLINE float one(const T* p) noexcept { return *p; }
    static SIMD_ML_ALWAYS_INLINE __m256 eight(const T* p) noexcept { return _mm256_loadu_ps(p); }
};
/** IEEE binary16: F16C vcvtph2ps (exact; subnormals, Inf and NaN included). */
struct F16 {
    using T = std::uint16_t;
    static SIMD_ML_ALWAYS_INLINE float one(const T* p) noexcept { return _cvtsh_ss(*p); }
    static SIMD_ML_ALWAYS_INLINE __m256 eight(const T* p) noexcept {
        return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    }
};
/** BF16 = the top 16 bits of an FP32: zero-extend and shift left 16 (exact). */
struct BF16 {
    using T = std::uint16_t;
    static SIMD_ML_ALWAYS_INLINE float one(const T* p) noexcept {
        const std::uint32_t bits = static_cast<std::uint32_t>(*p) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
    static SIMD_ML_ALWAYS_INLINE __m256 eight(const T* p) noexcept {
        const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
    }
};

/** uint8 minus a zero point (INT8 conversion path); exact in FP32, |a − zp| <= 255. */
struct U8 {
    using T = std::uint8_t;
    float zp = 0.0f;
    SIMD_ML_ALWAYS_INLINE float one(const T* p) const noexcept { return static_cast<float>(*p) - zp; }
    SIMD_ML_ALWAYS_INLINE __m256 eight(const T* p) const noexcept {
        const __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
        return _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v)), _mm256_set1_ps(zp));
    }
};
/** int8 (INT8 conversion path); exact in FP32. */
struct S8 {
    using T = std::int8_t;
    static SIMD_ML_ALWAYS_INLINE float one(const T* p) noexcept { return static_cast<float>(*p); }
    static SIMD_ML_ALWAYS_INLINE __m256 eight(const T* p) noexcept {
        const __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
        return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
    }
};

// ─── Micro-kernel ────────────────────────────────────────────────────────────
/**
 * kernel_6x16<R, Tail, Packed> — C[0..R)[0..n) += alpha · Σ_p A(r, p) · B(p, 0..n)
 *
 * A(r, p) = A[r*a_rs + p*a_cs]: packed micro-panel (a_rs = 1, a_cs = 6) or A in
 * place (a_rs = lda, a_cs = 1). B row p = B + p*b_rs: packed (b_rs = 16) or in
 * place (b_rs = ldb). Packed = true fixes those strides at compile time: with
 * them as run-time values the packed path measured 77-85% of OpenBLAS instead
 * of 98% (extra address arithmetic in the k-loop). R < 6 handles the last rows
 * of M; Tail (n < 16) masks the B loads and the C accesses so nothing past
 * column n is touched. EB is the element type of B in place (direct path of
 * the FP16/BF16 GEMMs): each B row segment is converted in registers when it
 * is loaded; packed B is always FP32.
 */
template <int R, bool Tail, bool Packed, class EB = F32>
SIMD_ML_ALWAYS_INLINE void kernel_6x16(int k, const float* __restrict__ A, std::ptrdiff_t a_rs_,
                                       std::ptrdiff_t a_cs_, const typename EB::T* __restrict__ B,
                                       std::ptrdiff_t b_rs_, float* __restrict__ C,
                                       std::ptrdiff_t ldc, int n, float alpha) noexcept {
    static_assert(!Packed || std::is_same_v<EB, F32>, "packed panels are FP32");
    const std::ptrdiff_t a_rs = Packed ? 1 : a_rs_;
    const std::ptrdiff_t a_cs = Packed ? MR : a_cs_;
    const std::ptrdiff_t b_rs = Packed ? NR : b_rs_;
    __m256 acc[R][2];
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_ps();
    __m256i m0 = _mm256_setzero_si256(), m1 = _mm256_setzero_si256();
    if (Tail) {
        const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        m0 = _mm256_cmpgt_epi32(_mm256_set1_epi32(n), lane);
        m1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(n - 8), lane);
    }
#pragma GCC unroll 4
    for (int p = 0; p < k; ++p) {
        const typename EB::T* b = B + p * b_rs;
        __m256 b0, b1;
        if constexpr (std::is_same_v<EB, F32>) {
            b0 = Tail ? _mm256_maskload_ps(b, m0) : _mm256_loadu_ps(b);
            b1 = Tail ? _mm256_maskload_ps(b + 8, m1) : _mm256_loadu_ps(b + 8);
        } else if (Tail) {  // AVX2 has no 16-bit masked load: copy the n values
            typename EB::T t[NR] = {};
            std::memcpy(t, b, static_cast<std::size_t>(n) * sizeof(t[0]));
            b0 = EB::eight(t);
            b1 = EB::eight(t + 8);
        } else {
            b0 = EB::eight(b);
            b1 = EB::eight(b + 8);
        }
        const float* a = A + p * a_cs;
        for (int r = 0; r < R; ++r) {
            const __m256 av = _mm256_broadcast_ss(a + r * a_rs);
            acc[r][0] = _mm256_fmadd_ps(av, b0, acc[r][0]);
            acc[r][1] = _mm256_fmadd_ps(av, b1, acc[r][1]);
        }
    }
    const __m256 alpha_v = _mm256_set1_ps(alpha);
    for (int r = 0; r < R; ++r) {
        float* c = C + r * ldc;
        if (Tail) {
            _mm256_maskstore_ps(c, m0, _mm256_fmadd_ps(acc[r][0], alpha_v, _mm256_maskload_ps(c, m0)));
            _mm256_maskstore_ps(c + 8, m1, _mm256_fmadd_ps(acc[r][1], alpha_v, _mm256_maskload_ps(c + 8, m1)));
        } else {
            _mm256_storeu_ps(c, _mm256_fmadd_ps(acc[r][0], alpha_v, _mm256_loadu_ps(c)));
            _mm256_storeu_ps(c + 8, _mm256_fmadd_ps(acc[r][1], alpha_v, _mm256_loadu_ps(c + 8)));
        }
    }
}

template <bool Tail, bool Packed, class EB = F32>
void run_tile_rows(int rows, int k, const float* A, std::ptrdiff_t a_rs, std::ptrdiff_t a_cs,
                   const typename EB::T* B, std::ptrdiff_t b_rs, float* C, std::ptrdiff_t ldc, int n,
                   float alpha) noexcept {
    switch (rows) {
        case 6: kernel_6x16<6, Tail, Packed, EB>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
        case 5: kernel_6x16<5, Tail, Packed, EB>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
        case 4: kernel_6x16<4, Tail, Packed, EB>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
        case 3: kernel_6x16<3, Tail, Packed, EB>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
        case 2: kernel_6x16<2, Tail, Packed, EB>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
        case 1: kernel_6x16<1, Tail, Packed, EB>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
        default: break;
    }
}

/** One micro-tile of packed micro-panels. */
SIMD_ML_ALWAYS_INLINE void run_tile_packed(int rows, int n, int k, const float* Ap,
                                           const float* Bp, float* C, std::ptrdiff_t ldc,
                                           float alpha) noexcept {
    if (rows == MR && n == NR)
        kernel_6x16<MR, false, true>(k, Ap, 1, MR, Bp, NR, C, ldc, n, alpha);
    else if (n == NR)
        run_tile_rows<false, true>(rows, k, Ap, 1, MR, Bp, NR, C, ldc, n, alpha);
    else
        run_tile_rows<true, true>(rows, k, Ap, 1, MR, Bp, NR, C, ldc, n, alpha);
}

// ─── Packing ─────────────────────────────────────────────────────────────────
/**
 * pack_a — op(A)[0..mc)[0..kc) into 6-row micro-panels, k-major:
 * Ap[(i/6)*6*kc + p*6 + r] = op(A)(i + r, p). Rows past mc are zero, so the
 * kernel's broadcasts read 6 consecutive floats per k-step.
 * op(A)(i, p) = A[i*lda + p], or A[p*lda + i] when trans (A stored K×M): then
 * the 6 values of one k-step are adjacent in memory and the copy is cheaper.
 * E converts FP16/BF16 to FP32 here, once per element of the MC×KC block.
 */
template <class E>
void pack_a(const E& e, const typename E::T* A, std::ptrdiff_t lda, bool trans, int mc, int kc, float* Ap) noexcept {
    for (int i = 0; i < mc; i += MR) {
        const int rows = std::min(MR, mc - i);
        float* dst = Ap + static_cast<std::ptrdiff_t>(i) * kc;
        if (trans) {
            const typename E::T* src = A + i;
            for (int p = 0; p < kc; ++p)
                for (int r = 0; r < MR; ++r) dst[p * MR + r] = r < rows ? e.one(src + p * lda + r) : 0.0f;
        } else if (rows == MR) {
            const typename E::T* src = A + i * lda;
            for (int p = 0; p < kc; ++p)
                for (int r = 0; r < MR; ++r) dst[p * MR + r] = e.one(src + r * lda + p);
        } else {
            const typename E::T* src = A + i * lda;
            for (int p = 0; p < kc; ++p)
                for (int r = 0; r < MR; ++r) dst[p * MR + r] = r < rows ? e.one(src + r * lda + p) : 0.0f;
        }
    }
}

/** transpose8x8 — r[q] becomes column q of the 8×8 block held in r[0..7]. */
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
 * pack_b — op(B)[0..kc)[0..nc) into 16-column micro-panels:
 * Bp[(j/16)*16*kc + p*16 + c] = op(B)(p, j + c). Columns past nc are zero.
 * op(B)(p, j) = B[p*ldb + j], or B[j*ldb + p] when trans (B stored N×K, as a
 * weight matrix W in x·Wᵀ): each column of the panel is then one row of B.
 * Full panels are transposed 8×8 in registers (8 rows of B × 8 values of p
 * per step); with a scalar gather the transposed packed path reached 59-86%
 * of OpenBLAS for M = 16..96 against 90-104% without the transpose.
 * E converts FP16/BF16 to FP32 in registers (8 values per vcvtph2ps, or per
 * zero-extend + shift for BF16), once per element of the KC×NC panel.
 */
template <class E>
void pack_b(const E& e, const typename E::T* B, std::ptrdiff_t ldb, bool trans, int kc, int nc, float* Bp) noexcept {
    for (int j = 0; j < nc; j += NR) {
        const int cols = std::min(NR, nc - j);
        float* dst = Bp + static_cast<std::ptrdiff_t>(j) * kc;
        if (trans) {
            int p0 = 0;
            if (cols == NR) {
                for (; p0 + 8 <= kc; p0 += 8)
                    for (int h = 0; h < NR; h += 8) {
                        __m256 r[8];
                        for (int q = 0; q < 8; ++q) r[q] = e.eight(B + (j + h + q) * ldb + p0);
                        transpose8x8(r);
                        for (int q = 0; q < 8; ++q) _mm256_storeu_ps(dst + (p0 + q) * NR + h, r[q]);
                    }
            }
            for (int c = 0; c < NR; ++c) {   // K tail, and partial panels
                const typename E::T* src = B + (j + c) * ldb;
                for (int p = p0; p < kc; ++p) dst[p * NR + c] = c < cols ? e.one(src + p) : 0.0f;
            }
        } else if (cols == NR) {
            const typename E::T* src = B + j;
            for (int p = 0; p < kc; ++p) {
                _mm256_storeu_ps(dst + p * NR, e.eight(src + p * ldb));
                _mm256_storeu_ps(dst + p * NR + 8, e.eight(src + p * ldb + 8));
            }
        } else {
            const typename E::T* src = B + j;
            for (int p = 0; p < kc; ++p)
                for (int c = 0; c < NR; ++c) dst[p * NR + c] = c < cols ? e.one(src + p * ldb + c) : 0.0f;
        }
    }
}

// ─── trans_b with few rows: dot products ─────────────────────────────────────
inline float hsum8(__m256 v) noexcept {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehdup_ps(t)));
}

/** The last n < 16 values of a row of B from b, as two FP32 vectors (zero past n). */
template <class EB>
SIMD_ML_ALWAYS_INLINE void load_tail16(const typename EB::T* b, int n, __m256i m0, __m256i m1,
                                       __m256& v0, __m256& v1) noexcept {
    if constexpr (std::is_same_v<EB, F32>) {
        v0 = _mm256_maskload_ps(b, m0);
        v1 = _mm256_maskload_ps(b + 8, m1);
    } else {  // AVX2 has no 16-bit masked load: copy the n values
        typename EB::T t[NR] = {};
        std::memcpy(t, b, static_cast<std::size_t>(n) * sizeof(t[0]));
        v0 = EB::eight(t);
        v1 = EB::eight(t + 8);
    }
}

/**
 * dot_rows<EB, J> — C[i][j..j+J) += alpha · Σ_p a[p] · B[j+q][p] for J <= 8
 * rows of B (stored N×K, elements EB::T): x·Wᵀ for one row x (FP32). This is
 * the decode GEMV: each element of B is used once, so the loop is bound by
 * reading B from DRAM and its job is to keep enough of it in flight. J rows
 * are J sequential streams for the hardware prefetchers, 16 values per row
 * per step; up to 4 rows have 2 accumulators each (8 for J = 4: enough to
 * cover the 4-cycle FMA latency with room for late loads), more rows one
 * each (16 YMM registers). FP16/BF16 rows are converted in registers (one
 * vcvtph2ps, or zero-extend + shift, per 8 values), so B is read at 2 bytes
 * per element. K tail: masked loads (FP32) or a copied tail.
 * Dev runs (Lunar Lake P-core, 1×4096×4096): 31-34 GB/s for FP32 and FP16,
 * about the 31-33 GB/s of a 4-stream read of a large array on that core
 * (docs/DESIGN.md §2.7).
 */
template <class EB, int J>
SIMD_ML_ALWAYS_INLINE void dot_rows(int K, const float* __restrict__ a, const typename EB::T* __restrict__ B,
                                    std::ptrdiff_t ldb, float* __restrict__ c, float alpha) noexcept {
    // Up to 4 rows: 2 accumulators per row; more rows: one (16 YMM registers).
    constexpr int U = J > 4 ? 1 : 2;
    __m256 acc[J][U];
    for (int q = 0; q < J; ++q)
        for (int u = 0; u < U; ++u) acc[q][u] = _mm256_setzero_ps();
    int p = 0;
    for (; p + 16 <= K; p += 16) {
        const __m256 a0 = _mm256_loadu_ps(a + p), a1 = _mm256_loadu_ps(a + p + 8);
        for (int q = 0; q < J; ++q) {
            const typename EB::T* b = B + q * ldb + p;
            acc[q][0] = _mm256_fmadd_ps(a0, EB::eight(b), acc[q][0]);
            acc[q][U - 1] = _mm256_fmadd_ps(a1, EB::eight(b + 8), acc[q][U - 1]);
        }
    }
    if (p < K) {
        const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i m0 = _mm256_cmpgt_epi32(_mm256_set1_epi32(K - p), lane);
        const __m256i m1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(K - p - 8), lane);
        const __m256 a0 = _mm256_maskload_ps(a + p, m0), a1 = _mm256_maskload_ps(a + p + 8, m1);
        for (int q = 0; q < J; ++q) {
            __m256 b0, b1;
            load_tail16<EB>(B + q * ldb + p, K - p, m0, m1, b0, b1);
            acc[q][0] = _mm256_fmadd_ps(a0, b0, acc[q][0]);
            acc[q][U - 1] = _mm256_fmadd_ps(a1, b1, acc[q][U - 1]);
        }
    }
    for (int q = 0; q < J; ++q)
        c[q] += alpha * hsum8(U == 2 ? _mm256_add_ps(acc[q][0], acc[q][U - 1]) : acc[q][0]);
}

/**
 * C (prescaled by beta) += alpha · A · Bᵀ, A FP32, B stored N×K (EB::T), row
 * by row: each block of R rows of B is read from memory once and reused from
 * L1/L2 for every row of A (R·K elements: 64 KiB at K = 4096 in FP32).
 * R = 4 rows for FP32, 8 for FP16/BF16: a 16-bit row step reads 32 bytes, half
 * of FP32's, so 8 streams keep as many bytes in flight as 4 FP32 ones. Dev
 * runs at 1×4096×4096 FP16 (Lunar Lake P-core, alternating builds): 4 rows
 * 28.1-30.8 GB/s, 8 rows 31.2-33.8 GB/s, FP32 31.6-34.3 GB/s in the same runs.
 * Software prefetch 256-1024 bytes ahead on every row changed nothing beyond
 * run-to-run noise (±2 GB/s), so the hardware prefetchers are left to it.
 */
template <class EB>
void dot_trans_b_rows(int M, int j0, int j1, int K, float alpha, const float* A, std::ptrdiff_t lda,
                      const typename EB::T* B, std::ptrdiff_t ldb, float* C, std::ptrdiff_t ldc) noexcept {
    constexpr int R = sizeof(typename EB::T) == 2 ? 8 : 4;
    static_assert(kGemvRowGroup % R == 0, "thread slices must start on a row block");
    const int N = j1;
    for (int j = j0; j < N; j += R) {
        const typename EB::T* Bj = B + j * ldb;
        for (int i = 0; i < M; ++i) {
            const float* a = A + i * lda;
            float* c = C + i * ldc + j;
            switch (std::min(R, N - j)) {
                case 8: if constexpr (R >= 8) dot_rows<EB, 8>(K, a, Bj, ldb, c, alpha); break;
                case 7: if constexpr (R >= 7) dot_rows<EB, 7>(K, a, Bj, ldb, c, alpha); break;
                case 6: if constexpr (R >= 6) dot_rows<EB, 6>(K, a, Bj, ldb, c, alpha); break;
                case 5: if constexpr (R >= 5) dot_rows<EB, 5>(K, a, Bj, ldb, c, alpha); break;
                case 4: dot_rows<EB, 4>(K, a, Bj, ldb, c, alpha); break;
                case 3: dot_rows<EB, 3>(K, a, Bj, ldb, c, alpha); break;
                case 2: dot_rows<EB, 2>(K, a, Bj, ldb, c, alpha); break;
                default: dot_rows<EB, 1>(K, a, Bj, ldb, c, alpha); break;
            }
        }
    }
}

/**
 * gemm_dot_trans_b — dot_trans_b_rows over all N rows of B; in OpenMP builds
 * split into one contiguous slice of rows per thread (gemv_threads /
 * gemv_slice): the A rows are shared read-only, each thread writes its own
 * columns of C, and every element is computed as with one thread.
 */
template <class EB>
void gemm_dot_trans_b(int M, int N, int K, float alpha, const float* A, std::ptrdiff_t lda,
                      const typename EB::T* B, std::ptrdiff_t ldb, float* C, std::ptrdiff_t ldc) noexcept {
#ifdef SIMD_ML_OPENMP
    const double bytes = static_cast<double>(N) * K * sizeof(typename EB::T);
    if (const int threads = gemv_threads(N, bytes); threads > 1) {
#pragma omp parallel num_threads(threads)
        {
            const GemvSlice s = gemv_slice(N, omp_get_thread_num(), omp_get_num_threads());
            dot_trans_b_rows<EB>(M, s.lo, s.hi, K, alpha, A, lda, B, ldb, C, ldc);
        }
        return;
    }
#endif
    dot_trans_b_rows<EB>(M, 0, N, K, alpha, A, lda, B, ldb, C, ldc);
}

// ─── Block sizes and buffers ─────────────────────────────────────────────────
struct Blocking { int kc, mc, nc; };

const Blocking& blocking() noexcept {
    static const Blocking b = []() noexcept {
        const dispatch::GemmBlocking g = dispatch::gemm_blocking(MR, NR);
        return Blocking{g.kc, g.mc, g.nc};
    }();
    return b;
}

/**
 * Per-thread packing buffers, grown on demand and kept for the thread's
 * lifetime (at most (MC+6)·KC + KC·(NC+16) floats: 4.8 MiB with the Lunar
 * Lake blocking; a threaded call's A buffer holds one MC×KC block per thread).
 * Allocating them per call cost 34 µs (2.1 MiB, WSL2/glibc).
 */
struct PackBuffers {
    AlignedUniquePtr<float> a, b;
    std::size_t a_cap = 0, b_cap = 0;
    float* get_a(std::size_t n) {
        if (n > a_cap) { a = make_aligned_array<float>(n); a_cap = n; }
        return a.get();
    }
    float* get_b(std::size_t n) {
        if (n > b_cap) { b = make_aligned_array<float>(n); b_cap = n; }
        return b.get();
    }
};

PackBuffers& thread_buffers() {
    thread_local PackBuffers buf;
    return buf;
}

/**
 * Operands as the loops see them: op(A) is M×K with elements EA::T, op(B) is
 * K×N with elements EB::T; ea / eb convert them to FP32 while packing (the
 * INT8 conversion path carries A's zero point in ea).
 */
template <class EA, class EB = EA>
struct Operands {
    const typename EA::T* A; std::ptrdiff_t lda; bool ta;
    const typename EB::T* B; std::ptrdiff_t ldb; bool tb;
    EA ea{};
    EB eb{};
    /** &op(A)(i, p) and &op(B)(p, j) */
    const typename EA::T* a_at(int i, int p) const { return ta ? A + p * lda + i : A + i * lda + p; }
    const typename EB::T* b_at(int p, int j) const { return tb ? B + j * ldb + p : B + p * ldb + j; }
};

/** Loops 2-5 for the column block op(B)[:, jc:jc+nc) (loop 1 is the caller). */
template <class EA, class EB>
void gemm_column_block(int M, int jc, int nc, int K, float alpha, const Operands<EA, EB>& op,
                       float* C, std::ptrdiff_t ldc, const Blocking& bl) {
    PackBuffers& buf = thread_buffers();
    const int kc_max = std::min(K, bl.kc);
    const int mc_max = std::min(M, bl.mc);
    float* Bp = buf.get_b(static_cast<std::size_t>(kc_max) * ((nc + NR - 1) / NR * NR));
    float* Ap = buf.get_a(static_cast<std::size_t>(kc_max) * ((mc_max + MR - 1) / MR * MR));

    for (int pc = 0; pc < K; pc += bl.kc) {                       // loop 2
        const int kc = std::min(K - pc, bl.kc);
        pack_b<EB>(op.eb, op.b_at(pc, jc), op.ldb, op.tb, kc, nc, Bp);
        for (int ic = 0; ic < M; ic += bl.mc) {                   // loop 3
            const int mc = std::min(M - ic, bl.mc);
            pack_a<EA>(op.ea, op.a_at(ic, pc), op.lda, op.ta, mc, kc, Ap);
            for (int jr = 0; jr < nc; jr += NR) {                 // loop 4
                const int n = std::min(NR, nc - jr);
                const float* Bm = Bp + static_cast<std::ptrdiff_t>(jr) * kc;
                for (int ir = 0; ir < mc; ir += MR) {             // loop 5
                    run_tile_packed(std::min(MR, mc - ir), n, kc,
                                    Ap + static_cast<std::ptrdiff_t>(ir) * kc, Bm,
                                    C + (ic + ir) * ldc + jc + jr, ldc, alpha);
                }
            }
        }
    }
}

#ifdef SIMD_ML_OPENMP
/**
 * Loops 2-5 for the column block op(B)[:, jc:jc+nc), run by every thread of
 * an OpenMP team (gemm_packed_team). Per KC panel: the team packs Bp together
 * (each thread a contiguous run of 16-column micro-panels) and waits; then
 * the chunks of the block (gemm_chunks: loop 3 cut into mc-row pieces, loop 4
 * into nc-column pieces) go to whichever thread is free, each packing the
 * chunk's rows of A into its own Ap. The barrier at the end of the omp for
 * keeps Bp until every chunk is done.
 */
template <class EA, class EB>
void gemm_column_block_team(int M, int jc, int nc, int K, float alpha, const Operands<EA, EB>& op,
                            float* C, std::ptrdiff_t ldc, const Blocking& bl, float* Bp,
                            float* Ap) noexcept {
    const int team = omp_get_num_threads(), t = omp_get_thread_num();
    const GemmChunks ch = gemm_chunks(team, M, nc, MR, NR, bl.mc);
    const int chunks = ch.m_chunks * ch.n_chunks;
    const int panels = (nc + NR - 1) / NR;
    const int b0 = static_cast<int>(static_cast<long long>(t) * panels / team) * NR;
    const int b1 = std::min(nc, static_cast<int>(static_cast<long long>(t + 1) * panels / team) * NR);
    for (int pc = 0; pc < K; pc += bl.kc) {                       // loop 2
        const int kc = std::min(K - pc, bl.kc);
        if (b1 > b0) pack_b<EB>(op.eb, op.b_at(pc, jc + b0), op.ldb, op.tb, kc, b1 - b0, Bp + static_cast<std::ptrdiff_t>(b0) * kc);
        int packed_ic = -1;                                       // rows now in this thread's Ap
#pragma omp barrier
#pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < chunks; ++c) {                        // loops 3 and 4, in chunks
            const int ic = c / ch.n_chunks * ch.mc, j0 = c % ch.n_chunks * ch.nc;
            const int mc = std::min(M - ic, ch.mc), j1 = std::min(nc, j0 + ch.nc);
            if (ic != packed_ic) { pack_a<EA>(op.ea, op.a_at(ic, pc), op.lda, op.ta, mc, kc, Ap); packed_ic = ic; }
            for (int jr = j0; jr < j1; jr += NR) {
                const int n = std::min(NR, nc - jr);
                const float* Bm = Bp + static_cast<std::ptrdiff_t>(jr) * kc;
                for (int ir = 0; ir < mc; ir += MR)               // loop 5
                    run_tile_packed(std::min(MR, mc - ir), n, kc,
                                    Ap + static_cast<std::ptrdiff_t>(ir) * kc, Bm,
                                    C + (ic + ir) * ldc + jc + jr, ldc, alpha);
            }
        }
    }
}

/**
 * Loops 1-5 on `threads` OpenMP threads. Buffers come from the calling
 * thread's PackBuffers, allocated before the parallel region (an exception
 * cannot leave one): one shared B panel, one A block per thread, each
 * rounded up to 64 bytes so two threads never write the same cache line.
 */
template <class EA, class EB>
void gemm_packed_team(int threads, int M, int N, int K, float alpha, const Operands<EA, EB>& op,
                      float* C, std::ptrdiff_t ldc, const Blocking& bl) {
    PackBuffers& buf = thread_buffers();
    const int kc_max = std::min(K, bl.kc);
    const std::size_t nc_max = static_cast<std::size_t>(std::min(N, bl.nc) + NR - 1) / NR * NR;
    const std::size_t mc_max = static_cast<std::size_t>(std::min(M, bl.mc) + MR - 1) / MR * MR;
    const std::size_t a_slice = (kc_max * mc_max + 15) / 16 * 16;
    float* Bp = buf.get_b(kc_max * nc_max);
    float* Ap = buf.get_a(a_slice * threads);
#pragma omp parallel num_threads(threads)
    {
        float* Ap_t = Ap + a_slice * omp_get_thread_num();
        for (int jc = 0; jc < N; jc += bl.nc)                     // loop 1
            gemm_column_block_team(M, jc, std::min(N - jc, bl.nc), K, alpha, op, C, ldc, bl, Bp, Ap_t);
    }
}
#endif

/** One strip of <= 6 rows of the direct path: C[0..rows) += alpha · A-strip · B, B in place. */
template <class EB>
void direct_rows(int rows, int N, int K, float alpha, const float* Ai, std::ptrdiff_t a_rs,
                 std::ptrdiff_t a_cs, const typename EB::T* B, std::ptrdiff_t ldb, float* Ci,
                 std::ptrdiff_t ldc) noexcept {
    for (int j = 0; j < N; j += NR) {
        const int n = std::min(NR, N - j);
        float* Cij = Ci + j;
        if (rows == MR && n == NR)
            kernel_6x16<MR, false, false, EB>(K, Ai, a_rs, a_cs, B + j, ldb, Cij, ldc, n, alpha);
        else if (n == NR)
            run_tile_rows<false, false, EB>(rows, K, Ai, a_rs, a_cs, B + j, ldb, Cij, ldc, n, alpha);
        else
            run_tile_rows<true, false, EB>(rows, K, Ai, a_rs, a_cs, B + j, ldb, Cij, ldc, n, alpha);
    }
}

/** Direct path on op(A) in place (any trans_a) and B in place (no trans_b). */
void direct(bool ta, int M, int N, int K, float alpha, const float* A, std::ptrdiff_t lda,
            const float* B, std::ptrdiff_t ldb, float* C, std::ptrdiff_t ldc) noexcept {
    const std::ptrdiff_t a_rs = ta ? 1 : lda, a_cs = ta ? lda : 1;
    for (int i = 0; i < M; i += MR)
        direct_rows<F32>(std::min(MR, M - i), N, K, alpha, ta ? A + i : A + i * lda, a_rs, a_cs,
                         B, ldb, C + i * ldc, ldc);
}

/** dst[0..n) = E-converted src[0..n). */
template <class E>
void convert_row(const typename E::T* src, int n, float* dst) noexcept {
    int p = 0;
    for (; p + 8 <= n; p += 8) _mm256_storeu_ps(dst + p, E::eight(src + p));
    for (; p < n; ++p) dst[p] = E::one(src + p);
}

/**
 * Direct path of the FP16/BF16 GEMMs (same shapes as FP32's). Each 6-row
 * strip of A is converted once into a 6×K float buffer. B is used by every
 * strip: with one strip (M <= 6, e.g. M = 1) it is read in place and converted
 * in registers, so it is read once at 2 bytes per element; with more strips
 * it is converted once into a K×N float buffer (up to 4 MiB; larger B stays
 * in place). Converting in the kernel for every strip ran BF16 at 65% of FP32
 * at N = 64 (zero-extend + shift on every load; Lunar Lake P-core).
 * Cost: with N <= 16 (one panel) and large M·K, the FP32 kernel issues one A
 * broadcast per 2 FMAs, so converting and storing every A element first is a
 * large share of the work: 2048×16×2048 and 4096×8×4096 run at 70-82% of the
 * FP32 GEMM (converting in chunks of 512 along K, to keep the strip in L1D,
 * did not change that). Single-row and few-row shapes read half the bytes of
 * FP32 and run 1.4-2.9× faster (1×4096×4096, 6-7×4096×1024).
 */
template <class E>
void direct_lowp(int M, int N, int K, float alpha, const typename E::T* A, std::ptrdiff_t lda,
                 const typename E::T* B, std::ptrdiff_t ldb, float* C, std::ptrdiff_t ldc) {
    PackBuffers& buf = thread_buffers();
    float* As = buf.get_a(static_cast<std::size_t>(MR) * K);
    const bool convert_b = M > MR && static_cast<long long>(K) * N <= (1LL << 20);
    float* Bf = convert_b ? buf.get_b(static_cast<std::size_t>(K) * N) : nullptr;
    if (convert_b)
        for (int p = 0; p < K; ++p) convert_row<E>(B + p * ldb, N, Bf + static_cast<std::ptrdiff_t>(p) * N);
    for (int i = 0; i < M; i += MR) {
        const int rows = std::min(MR, M - i);
        for (int r = 0; r < rows; ++r) convert_row<E>(A + (i + r) * lda, K, As + r * K);
        if (convert_b)
            direct_rows<F32>(rows, N, K, alpha, As, K, 1, Bf, N, C + i * ldc, ldc);
        else
            direct_rows<E>(rows, N, K, alpha, As, K, 1, B, ldb, C + i * ldc, ldc);
    }
}

/** Loops 1-5 (C already scaled by beta, K > 0). */
template <class EA, class EB>
void packed(int M, int N, int K, float alpha, const Operands<EA, EB>& op, float* C, std::ptrdiff_t ldc) {
    const Blocking& bl = blocking();
#ifdef SIMD_ML_OPENMP
    if (const int threads = gemm_threads(M, N, K, MR, NR); threads > 1) {
        gemm_packed_team(threads, M, N, K, alpha, op, C, ldc, bl);
        return;
    }
#endif
    for (int jc = 0; jc < N; jc += bl.nc)                         // loop 1
        gemm_column_block(M, jc, std::min(N - jc, bl.nc), K, alpha, op, C, ldc, bl);
}

/**
 * FP16/BF16 inputs, FP32 accumulation and output: the FP32 paths with the
 * conversion moved into packing (or, on the direct path, into the A-strip
 * buffer and the B loads). Replaces a one-accumulator loop that converted B on
 * every use and had no cache blocking (12-17% of the FP32 GEMM at N = 256-1024
 * on the reference laptop).
 */
/**
 * trans_b with few rows (x·Wᵀ, decode), FP16/BF16: the rows of A (at most
 * kDotRowsMax·K values) are converted to FP32 once, then the row-streaming
 * GEMV reads B in place and converts it in registers.
 */
template <class E>
void dot_lowp(int M, int N, int K, float alpha, const typename E::T* A, std::ptrdiff_t lda,
              const typename E::T* B, std::ptrdiff_t ldb, float* C, std::ptrdiff_t ldc) {
    float* Af = thread_buffers().get_a(static_cast<std::size_t>(M) * K);
    for (int i = 0; i < M; ++i) convert_row<E>(A + i * lda, K, Af + static_cast<std::ptrdiff_t>(i) * K);
    gemm_dot_trans_b<E>(M, N, K, alpha, Af, K, B, ldb, C, ldc);
}

template <class E>
void sgemm_lowp(bool tb, int M, int N, int K, float alpha, const typename E::T* A, int lda,
                const typename E::T* B, int ldb, float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    scale_matrix_c(C, M, N, ldc, beta);
    if (K == 0) return;
    if (tb && M <= kDotRowsMax)
        dot_lowp<E>(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else if (!tb && sgemm_avx2_prefers_direct(M, N, K))
        direct_lowp<E>(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else
        packed(M, N, K, alpha, Operands<E>{A, lda, false, B, ldb, tb}, C, ldc);
}

enum class Path { direct, dot, packed };

Path choose_path(bool ta, bool tb, int M, int N, int K) noexcept {
    // trans_b without packing needs rows of B along K: the dot-product path,
    // for a few rows of A (x·Wᵀ with a small batch). Dot vs packed, % of
    // OpenBLAS with transb (Lunar Lake P-core): M=1 291/42, 4 169/39, 8 153/44
    // (×4096×1024); 16×1024×1024 90/70, 24×… 78/74, 32×… 75/77, 64×… 80/84;
    // 128³ 75/89. OpenBLAS's own transb GEMM is slow for M <= 8 (6 GFLOP/s at
    // M = 1), so the dot path is well ahead there.
    // Each row of A re-reads the current R-row block of B from cache, with 2
    // loads per FMA, so past a few rows packing wins. GFLOP/s dot / packed at
    // N = 4096 (dev runs, same core), K = 4096: FP32 M=8 34.0/32.8, M=12
    // 33.8/46.1; FP16 M=8 56/41, 12 57/56, 16 60/63; BF16 12 58/56, 16 58/61.
    // K = 1024: FP32 M=8 51/33, 12 52/47, 16 51/54; FP16 8 52/39, 12 49/57;
    // BF16 8 47/44, 12 48/54. At M = 1 the dot path is 3-5x faster. M <= 8
    // (kDotRowsMax) wins or ties in every case; above it the most packing
    // gives up is 9% (FP32, 12×1024×1024).
    if (tb) return (!ta && M <= kDotRowsMax) ? Path::dot : Path::packed;
    return sgemm_avx2_prefers_direct(M, N, K) ? Path::direct : Path::packed;
}

}  // namespace

void scale_matrix_c(float* C, int M, int N, int ldc, float beta) noexcept {
    if (beta == 1.0f) return;
    for (int i = 0; i < M; ++i) {
        float* row = C + static_cast<std::ptrdiff_t>(i) * ldc;
        if (beta == 0.0f) {
            std::fill(row, row + N, 0.0f);   // never multiply: C may hold NaN
        } else {
            for (int j = 0; j < N; ++j) row[j] *= beta;
        }
    }
}

void sgemm_blocking_avx2(int* kc, int* mc, int* nc) noexcept {
    const Blocking& b = blocking();
    *kc = b.kc; *mc = b.mc; *nc = b.nc;
}

bool sgemm_avx2_prefers_direct(int M, int N, int K) noexcept {
    // Packing copies M·K + K·N floats once per call. Both paths on the same
    // 6×16 kernel, % of OpenBLAS (Lunar Lake P-core, interleaved, median of 31):
    //   square 32: direct 97 / packed 89;  64: 93 / 86;  96: 81 / 90;
    //   128: 74 / 93 (direct loses most at power-of-two ldb: B rows 512 B or
    //   1 KiB apart fall into 8 or 4 of the 64 L1 sets);  256: 70 / 98;
    //   1024×16×1024: 101 / 79;  256×256×32: 76 / 96;  1×4096×1024: 76 / 54;
    //   6×4096×1024: 58 / 66;  7×4096×1024: 47 / 72;  48×1024×1024: 64 / 96.
    // So: direct for products up to ~80³, for N within one 16-wide panel, and
    // for a single row (M = 1, GEMV: packing B costs as much as using it).
    return M == 1 || N <= NR || static_cast<long long>(M) * N * K <= 80LL * 80 * 80;
}

void sgemm_direct_avx2(bool trans_a, bool trans_b, int M, int N, int K, float alpha,
                       const float* A, int lda, const float* B, int ldb, float beta,
                       float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    scale_matrix_c(C, M, N, ldc, beta);
    if (K == 0) return;
    if (trans_b && !trans_a)
        gemm_dot_trans_b<F32>(M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else if (!trans_b)
        direct(trans_a, M, N, K, alpha, A, lda, B, ldb, C, ldc);
    else  // both transposed: packing handles it
        sgemm_packed_avx2(trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, 1.0f, C, ldc);
}

void sgemm_packed_avx2(bool trans_a, bool trans_b, int M, int N, int K,
                       float alpha,
                       const float* A, int lda,
                       const float* B, int ldb,
                       float beta,
                       float* C, int ldc) {
    // K == 0 is valid (BLAS semantics): A·B is empty, so C = beta·C.
    if (M <= 0 || N <= 0 || K < 0) return;
    const Path path = choose_path(trans_a, trans_b, M, N, K);
    if (path != Path::packed) {
        sgemm_direct_avx2(trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    scale_matrix_c(C, M, N, ldc, beta);
    if (K == 0) return;
    packed(M, N, K, alpha, Operands<F32>{A, lda, trans_a, B, ldb, trans_b}, C, ldc);
}

void sgemm_f16_avx2(bool trans_b, int M, int N, int K, float alpha, const std::uint16_t* A, int lda,
                    const std::uint16_t* B, int ldb, float beta, float* C, int ldc) {
    sgemm_lowp<F16>(trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void sgemm_bf16_avx2(bool trans_b, int M, int N, int K, float alpha, const std::uint16_t* A, int lda,
                     const std::uint16_t* B, int ldb, float beta, float* C, int ldc) {
    sgemm_lowp<BF16>(trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

namespace {

/**
 * int8_rows.hpp step without VNNI: a − zp and b widened to int16, vpmaddwd
 * (exact; vpmaddubsw would saturate). For one row times B [N, K].
 */
struct StepInt16 {
    using Zp = __m256i;
    struct A { __m256i lo, hi; };
    static Zp zero_point(int zp) noexcept { return _mm256_set1_epi16(static_cast<short>(zp)); }
    static A load_a(const std::uint8_t* p, Zp z) noexcept {
        return {_mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))), z),
                _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16))), z)};
    }
    static SIMD_ML_ALWAYS_INLINE void step(const A& a, __m256i b, __m256i& ab, __m256i&) noexcept {
        const __m256i lo = _mm256_madd_epi16(a.lo, _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b)));
        const __m256i hi = _mm256_madd_epi16(a.hi, _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1)));
        ab = _mm256_add_epi32(ab, _mm256_add_epi32(lo, hi));
    }
    static long long finish(__m256i ab, __m256i, int) noexcept { return hsum_epi32_to_i64(ab); }
};

constexpr int kU8S8Rows = 4;

}  // namespace

/**
 * INT8 without VNNI: A − zero point and B are converted to FP32 while packing
 * (exact: |values| <= 255), the FP32 loops and 6×16 kernel compute the sums,
 * then each column is scaled by a_scale·b_scales[j]. The same work as the FP32
 * GEMM plus the conversion, so this is a memory saving (1-byte operands), not
 * a speed-up; the VNNI paths are the fast ones. One row with B [N, K]
 * (M == 1, trans_b) streams B in place with int16 dots instead (int8_rows.hpp).
 */
void sgemm_u8s8_convert_avx2(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                             int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                             const float* b_scales, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    if (M == 1 && trans_b) {   // one row x·Wᵀ: stream B in place (int8_rows.hpp)
        u8s8_rows<StepInt16, kU8S8Rows>(N, K, A, a_zero_point, B, ldb, a_scale, b_scales, C);
        return;
    }
    scale_matrix_c(C, M, N, ldc, 0.0f);
    if (K > 0)
        packed(M, N, K, 1.0f,
               Operands<U8, S8>{A, lda, false, B, ldb, trans_b, U8{static_cast<float>(a_zero_point)}, S8{}},
               C, ldc);
    const __m256 as = _mm256_set1_ps(a_scale);
    for (int i = 0; i < M; ++i) {
        float* c = C + static_cast<std::ptrdiff_t>(i) * ldc;
        int j = 0;
        for (; j + 8 <= N; j += 8)
            _mm256_storeu_ps(c + j, _mm256_mul_ps(_mm256_loadu_ps(c + j), _mm256_mul_ps(as, _mm256_loadu_ps(b_scales + j))));
        for (; j < N; ++j) c[j] *= a_scale * b_scales[j];
    }
}

void sgemm_u8s8_avx2(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                     int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                     const float* b_scales, float* C, int ldc) {
#ifdef SIMD_ML_HAVE_AVXVNNI
    if (avx_vnni_available()) {
        sgemm_u8s8_avxvnni(trans_b, M, N, K, A, lda, a_zero_point, B, ldb, a_scale, b_scales, C, ldc);
        return;
    }
#endif
    sgemm_u8s8_convert_avx2(trans_b, M, N, K, A, lda, a_zero_point, B, ldb, a_scale, b_scales, C, ldc);
}

}  // namespace gemm
}  // namespace simd_ml
