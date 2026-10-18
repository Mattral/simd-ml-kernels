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
#include "cache_alloc.hpp"
#include "../../dispatch/cpuid.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if !defined(__AVX2__) || !defined(__FMA__)
#  error "avx2_gemm_packed.cpp is an AVX2 unit: compile with -mavx2 -mfma (CMakeLists.txt)"
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
 * column n is touched.
 */
template <int R, bool Tail, bool Packed>
SIMD_ML_ALWAYS_INLINE void kernel_6x16(int k, const float* __restrict__ A, std::ptrdiff_t a_rs_,
                                       std::ptrdiff_t a_cs_, const float* __restrict__ B,
                                       std::ptrdiff_t b_rs_, float* __restrict__ C,
                                       std::ptrdiff_t ldc, int n, float alpha) noexcept {
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
        const float* b = B + p * b_rs;
        const __m256 b0 = Tail ? _mm256_maskload_ps(b, m0) : _mm256_loadu_ps(b);
        const __m256 b1 = Tail ? _mm256_maskload_ps(b + 8, m1) : _mm256_loadu_ps(b + 8);
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

template <bool Tail, bool Packed>
void run_tile_rows(int rows, int k, const float* A, std::ptrdiff_t a_rs, std::ptrdiff_t a_cs,
                   const float* B, std::ptrdiff_t b_rs, float* C, std::ptrdiff_t ldc, int n,
                   float alpha) noexcept {
    switch (rows) {
        case 6: kernel_6x16<6, Tail, Packed>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
        case 5: kernel_6x16<5, Tail, Packed>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
        case 4: kernel_6x16<4, Tail, Packed>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
        case 3: kernel_6x16<3, Tail, Packed>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
        case 2: kernel_6x16<2, Tail, Packed>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
        case 1: kernel_6x16<1, Tail, Packed>(k, A, a_rs, a_cs, B, b_rs, C, ldc, n, alpha); break;
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
 */
void pack_a(const float* A, std::ptrdiff_t lda, bool trans, int mc, int kc, float* Ap) noexcept {
    for (int i = 0; i < mc; i += MR) {
        const int rows = std::min(MR, mc - i);
        float* dst = Ap + static_cast<std::ptrdiff_t>(i) * kc;
        if (trans) {
            const float* src = A + i;
            for (int p = 0; p < kc; ++p)
                for (int r = 0; r < MR; ++r) dst[p * MR + r] = r < rows ? src[p * lda + r] : 0.0f;
        } else if (rows == MR) {
            const float* src = A + i * lda;
            for (int p = 0; p < kc; ++p)
                for (int r = 0; r < MR; ++r) dst[p * MR + r] = src[r * lda + p];
        } else {
            const float* src = A + i * lda;
            for (int p = 0; p < kc; ++p)
                for (int r = 0; r < MR; ++r) dst[p * MR + r] = r < rows ? src[r * lda + p] : 0.0f;
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
 */
void pack_b(const float* B, std::ptrdiff_t ldb, bool trans, int kc, int nc, float* Bp) noexcept {
    for (int j = 0; j < nc; j += NR) {
        const int cols = std::min(NR, nc - j);
        float* dst = Bp + static_cast<std::ptrdiff_t>(j) * kc;
        if (trans) {
            int p0 = 0;
            if (cols == NR) {
                for (; p0 + 8 <= kc; p0 += 8)
                    for (int h = 0; h < NR; h += 8) {
                        __m256 r[8];
                        for (int q = 0; q < 8; ++q) r[q] = _mm256_loadu_ps(B + (j + h + q) * ldb + p0);
                        transpose8x8(r);
                        for (int q = 0; q < 8; ++q) _mm256_storeu_ps(dst + (p0 + q) * NR + h, r[q]);
                    }
            }
            for (int c = 0; c < NR; ++c) {   // K tail, and partial panels
                const float* src = B + (j + c) * ldb;
                for (int p = p0; p < kc; ++p) dst[p * NR + c] = c < cols ? src[p] : 0.0f;
            }
        } else if (cols == NR) {
            const float* src = B + j;
            for (int p = 0; p < kc; ++p) {
                _mm256_storeu_ps(dst + p * NR, _mm256_loadu_ps(src + p * ldb));
                _mm256_storeu_ps(dst + p * NR + 8, _mm256_loadu_ps(src + p * ldb + 8));
            }
        } else {
            const float* src = B + j;
            for (int p = 0; p < kc; ++p)
                for (int c = 0; c < NR; ++c) dst[p * NR + c] = c < cols ? src[p * ldb + c] : 0.0f;
        }
    }
}

// ─── trans_b with few rows: dot products ─────────────────────────────────────
inline float hsum8(__m256 v) noexcept {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_movehdup_ps(t)));
}

/**
 * dot_rows<J> — C[i][j..j+J) += alpha · Σ_p A[i][p] · B[j+c][p] for J <= 4 rows
 * of B (stored N×K): x·Wᵀ for a few rows of x. Both operands are read along
 * contiguous rows, 16 floats per step into 2 accumulators per output (8 for
 * J = 4: enough to cover the FMA latency), masked loads for the K tail.
 */
template <int J>
SIMD_ML_ALWAYS_INLINE void dot_rows(int K, const float* __restrict__ a, const float* __restrict__ B,
                                    std::ptrdiff_t ldb, float* __restrict__ c, float alpha) noexcept {
    __m256 acc[J][2];
    for (int q = 0; q < J; ++q) acc[q][0] = acc[q][1] = _mm256_setzero_ps();
    int p = 0;
    for (; p + 16 <= K; p += 16) {
        const __m256 a0 = _mm256_loadu_ps(a + p), a1 = _mm256_loadu_ps(a + p + 8);
        for (int q = 0; q < J; ++q) {
            acc[q][0] = _mm256_fmadd_ps(a0, _mm256_loadu_ps(B + q * ldb + p), acc[q][0]);
            acc[q][1] = _mm256_fmadd_ps(a1, _mm256_loadu_ps(B + q * ldb + p + 8), acc[q][1]);
        }
    }
    if (p < K) {
        const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256i m0 = _mm256_cmpgt_epi32(_mm256_set1_epi32(K - p), lane);
        const __m256i m1 = _mm256_cmpgt_epi32(_mm256_set1_epi32(K - p - 8), lane);
        const __m256 a0 = _mm256_maskload_ps(a + p, m0), a1 = _mm256_maskload_ps(a + p + 8, m1);
        for (int q = 0; q < J; ++q) {
            acc[q][0] = _mm256_fmadd_ps(a0, _mm256_maskload_ps(B + q * ldb + p, m0), acc[q][0]);
            acc[q][1] = _mm256_fmadd_ps(a1, _mm256_maskload_ps(B + q * ldb + p + 8, m1), acc[q][1]);
        }
    }
    for (int q = 0; q < J; ++q) c[q] += alpha * hsum8(_mm256_add_ps(acc[q][0], acc[q][1]));
}

/** C (prescaled by beta) += alpha · A · Bᵀ, B stored N×K, row by row. */
void gemm_dot_trans_b(int M, int N, int K, float alpha, const float* A, std::ptrdiff_t lda,
                      const float* B, std::ptrdiff_t ldb, float* C, std::ptrdiff_t ldc) noexcept {
    for (int j = 0; j < N; j += 4) {          // 4 rows of B, reused for every row of A
        const float* Bj = B + j * ldb;
        for (int i = 0; i < M; ++i) {
            const float* a = A + i * lda;
            float* c = C + i * ldc + j;
            switch (std::min(4, N - j)) {
                case 4: dot_rows<4>(K, a, Bj, ldb, c, alpha); break;
                case 3: dot_rows<3>(K, a, Bj, ldb, c, alpha); break;
                case 2: dot_rows<2>(K, a, Bj, ldb, c, alpha); break;
                default: dot_rows<1>(K, a, Bj, ldb, c, alpha); break;
            }
        }
    }
}

// ─── Block sizes and buffers ─────────────────────────────────────────────────
struct Blocking { int kc, mc, nc; };

const Blocking& blocking() noexcept {
    static const Blocking b = [] {
        const dispatch::GemmBlocking g = dispatch::gemm_blocking(MR, NR);
        return Blocking{g.kc, g.mc, g.nc};
    }();
    return b;
}

/**
 * Per-thread packing buffers, grown on demand and kept for the thread's
 * lifetime (at most (MC+6)·KC + KC·(NC+16) floats: 4.8 MiB with the Lunar
 * Lake blocking). Allocating them per call cost 34 µs (2.1 MiB, WSL2/glibc).
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

/** Operands as the loops see them: op(A) is M×K, op(B) is K×N. */
struct Operands {
    const float* A; std::ptrdiff_t lda; bool ta;
    const float* B; std::ptrdiff_t ldb; bool tb;
    /** &op(A)(i, p) and &op(B)(p, j) */
    const float* a_at(int i, int p) const { return ta ? A + p * lda + i : A + i * lda + p; }
    const float* b_at(int p, int j) const { return tb ? B + j * ldb + p : B + p * ldb + j; }
};

/** Loops 2-5 for the column block op(B)[:, jc:jc+nc) (loop 1 is the caller). */
void gemm_column_block(int M, int jc, int nc, int K, float alpha, const Operands& op,
                       float* C, std::ptrdiff_t ldc, const Blocking& bl) {
    PackBuffers& buf = thread_buffers();
    const int kc_max = std::min(K, bl.kc);
    const int mc_max = std::min(M, bl.mc);
    float* Bp = buf.get_b(static_cast<std::size_t>(kc_max) * ((nc + NR - 1) / NR * NR));
    float* Ap = buf.get_a(static_cast<std::size_t>(kc_max) * ((mc_max + MR - 1) / MR * MR));

    for (int pc = 0; pc < K; pc += bl.kc) {                       // loop 2
        const int kc = std::min(K - pc, bl.kc);
        pack_b(op.b_at(pc, jc), op.ldb, op.tb, kc, nc, Bp);
        for (int ic = 0; ic < M; ic += bl.mc) {                   // loop 3
            const int mc = std::min(M - ic, bl.mc);
            pack_a(op.a_at(ic, pc), op.lda, op.ta, mc, kc, Ap);
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

/** Direct path on op(A) in place (any trans_a) and B in place (no trans_b). */
void direct(bool ta, int M, int N, int K, float alpha, const float* A, std::ptrdiff_t lda,
            const float* B, std::ptrdiff_t ldb, float* C, std::ptrdiff_t ldc) noexcept {
    const std::ptrdiff_t a_rs = ta ? 1 : lda, a_cs = ta ? lda : 1;
    for (int i = 0; i < M; i += MR) {
        const int rows = std::min(MR, M - i);
        const float* Ai = ta ? A + i : A + i * lda;
        for (int j = 0; j < N; j += NR) {
            const int n = std::min(NR, N - j);
            float* Cij = C + i * ldc + j;
            if (rows == MR && n == NR)
                kernel_6x16<MR, false, false>(K, Ai, a_rs, a_cs, B + j, ldb, Cij, ldc, n, alpha);
            else if (n == NR)
                run_tile_rows<false, false>(rows, K, Ai, a_rs, a_cs, B + j, ldb, Cij, ldc, n, alpha);
            else
                run_tile_rows<true, false>(rows, K, Ai, a_rs, a_cs, B + j, ldb, Cij, ldc, n, alpha);
        }
    }
}

enum class Path { direct, dot, packed };

Path choose_path(bool ta, bool tb, int M, int N, int K) noexcept {
    // trans_b without packing needs rows of B along K: the dot-product path,
    // for a few rows of A (x·Wᵀ with a small batch). Dot vs packed, % of
    // OpenBLAS with transb (Lunar Lake P-core): M=1 291/42, 4 169/39, 8 153/44
    // (×4096×1024); 16×1024×1024 90/70, 24×… 78/74, 32×… 75/77, 64×… 80/84;
    // 128³ 75/89. OpenBLAS's own transb GEMM is slow for M <= 8 (6 GFLOP/s at
    // M = 1), so the dot path is well ahead there.
    if (tb) return (!ta && M <= 16) ? Path::dot : Path::packed;
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
                       float* C, int ldc) noexcept {
    if (M <= 0 || N <= 0 || K < 0) return;
    scale_matrix_c(C, M, N, ldc, beta);
    if (K == 0) return;
    if (trans_b && !trans_a)
        gemm_dot_trans_b(M, N, K, alpha, A, lda, B, ldb, C, ldc);
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
    const Blocking& bl = blocking();
    const Operands op{A, lda, trans_a, B, ldb, trans_b};

    // Loop 1 (jc): NC-wide column blocks, one per OpenMP thread.
#ifdef SIMD_ML_OPENMP
#pragma omp parallel for schedule(dynamic, 1) \
    num_threads(get_num_threads()) if (get_num_threads() > 1)
#endif
    for (int jc = 0; jc < N; jc += bl.nc)
        gemm_column_block(M, jc, std::min(N - jc, bl.nc), K, alpha, op, C, ldc, bl);
}

}  // namespace gemm
}  // namespace simd_ml
