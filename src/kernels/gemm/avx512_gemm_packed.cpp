/**
 * avx512_gemm_packed.cpp — FP32 GEMM, AVX-512: a 12×32 micro-kernel (24 ZMM
 * accumulators) inside the same Goto/BLIS five loops as the AVX2 path
 * (avx2_gemm_packed.cpp, docs/DESIGN.md §2). An AVX-512 unit: compiled with
 * -mavx512f -mavx512dq -mavx512bw -mavx512vl and used only when dispatch.cpp
 * has found those and OS ZMM state. Small products (sgemm_avx2_prefers_direct)
 * use the AVX2 direct path, which any CPU that runs this unit also runs.
 *
 * ─── Why 12×32 (24 accumulators) ─────────────────────────────────────────────
 * Ports and latency: Skylake-SP to Sapphire Rapids (two 512-bit FMA units) and
 * Zen 5 issue 2 ZMM FMAs per cycle with 4-cycle latency, so 8 independent
 * accumulators are the minimum. The previous 8×16 kernel had exactly 8 and
 * left 23 of the 32 ZMM registers unused; any L1 miss or late broadcast
 * stalled both FMA pipes.
 * Registers: 12 rows × 2 vectors = 24 accumulators + 2 B vectors = 26 of 32.
 * A is broadcast straight from memory (vbroadcastss / embedded {1to16}), so
 * no register holds it. 14×32 (28 accumulators, 30 registers) also fits;
 * MR is a single constant below.
 * Loads per FMA: per k-step 2 B loads + 12 broadcasts feed 24 FMAs (0.58);
 * two load ports (Skylake-SP) need 7 cycles for them, the FMAs 12.
 * Shape: the row-major view of BLIS's Skylake-X SGEMM kernel (32×12,
 * column-major, 24 accumulators).
 * Performance of this kernel is measured on real AVX-512 hardware only (cloud
 * VM; docs/BENCHMARKS.md). Intel SDE runs check correctness, not speed.
 *
 * ─── Block sizes ──────────────────────────────────────────────────────────────
 * dispatch::gemm_blocking(12, 32): the KC×32 B micro-panel fills half of L1D
 * (KC = 128 with 32 KiB, 192 with 48 KiB), the MC×KC A block a quarter of L2,
 * the KC×NC B panel half of the LLC; same rules as AVX2.
 */

#include "avx2_gemm_packed.hpp"
#include "bf16_gemm.hpp"
#include "f16_gemm.hpp"
#include "int8_gemm.hpp"
#include "cache_alloc.hpp"
#include "../../dispatch/cpuid.hpp"

#include <algorithm>
#include <cstddef>
#include <array>
#include <cstdint>
#include <cstring>
#include <utility>

#if !defined(__AVX512F__)
#  error "avx512_gemm_packed.cpp is an AVX-512 unit: compile with -mavx512f (CMakeLists.txt)"
#endif
#include <immintrin.h>
#ifdef SIMD_ML_OPENMP
#  include <omp.h>
#endif

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

// ─── Input element types (as in the AVX2 unit) ───────────────────────────────
// Packing reads A and B through one of these and writes FP32; masked loads
// return 0 in the masked-off lanes, which converts to +0.0f.
struct F32 {
    using T = float;
    static SIMD_ML_ALWAYS_INLINE float one(const T* p) noexcept { return *p; }
    static SIMD_ML_ALWAYS_INLINE __m512 sixteen(__mmask16 m, const T* p) noexcept {
        return _mm512_maskz_loadu_ps(m, p);
    }
};
struct F16 {
    using T = std::uint16_t;
    static SIMD_ML_ALWAYS_INLINE float one(const T* p) noexcept { return _cvtsh_ss(*p); }
    static SIMD_ML_ALWAYS_INLINE __m512 sixteen(__mmask16 m, const T* p) noexcept {
        return _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(m, p));
    }
};
struct BF16 {
    using T = std::uint16_t;
    static SIMD_ML_ALWAYS_INLINE float one(const T* p) noexcept {
        const std::uint32_t bits = static_cast<std::uint32_t>(*p) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
    static SIMD_ML_ALWAYS_INLINE __m512 sixteen(__mmask16 m, const T* p) noexcept {
        const __m512i w = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(m, p));
        return _mm512_castsi512_ps(_mm512_slli_epi32(w, 16));
    }
};

/**
 * kernel_12x32<R, Tail> — C[0..R)[0..n) += alpha · Ap · Bp on packed
 * micro-panels: Ap[p*12 + r] = A(r, p), Bp[p*32 + c] = B(p, c). R < 12 handles
 * the last rows of M; Tail (n < 32) masks the C accesses (B is zero-padded).
 */
template <int R, bool Tail>
SIMD_ML_ALWAYS_INLINE void kernel_12x32(int k, const float* __restrict__ Ap,
                                        const float* __restrict__ Bp, float* __restrict__ C,
                                        std::ptrdiff_t ldc, int n, float alpha) noexcept {
    __m512 acc[R][2];
    for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = _mm512_setzero_ps();
#pragma GCC unroll 4
    for (int p = 0; p < k; ++p) {
        const __m512 b0 = _mm512_loadu_ps(Bp + p * NR);
        const __m512 b1 = _mm512_loadu_ps(Bp + p * NR + 16);
        const float* a = Ap + p * MR;
        for (int r = 0; r < R; ++r) {
            const __m512 av = _mm512_set1_ps(a[r]);
            acc[r][0] = _mm512_fmadd_ps(av, b0, acc[r][0]);
            acc[r][1] = _mm512_fmadd_ps(av, b1, acc[r][1]);
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

using KernelFn = void (*)(int, const float*, const float*, float*, std::ptrdiff_t, int, float);

template <bool Tail, std::size_t... I>
constexpr auto make_row_table(std::index_sequence<I...>) {
    // Entry r runs a kernel for r + 1 rows.
    return std::array<KernelFn, sizeof...(I)>{{&kernel_12x32<int(I) + 1, Tail>...}};
}

/** One micro-tile: rows <= 12, n <= 32 columns. */
SIMD_ML_ALWAYS_INLINE void run_tile(int rows, int n, int k, const float* Ap, const float* Bp,
                                    float* C, std::ptrdiff_t ldc, float alpha) noexcept {
    static constexpr auto full = make_row_table<false>(std::make_index_sequence<MR>{});
    static constexpr auto tail = make_row_table<true>(std::make_index_sequence<MR>{});
    if (rows == MR && n == NR) kernel_12x32<MR, false>(k, Ap, Bp, C, ldc, n, alpha);
    else (n == NR ? full : tail)[rows - 1](k, Ap, Bp, C, ldc, n, alpha);
}

/**
 * pack_a — op(A) into 12-row micro-panels, k-major, rows past mc zero; trans:
 * A stored K×M (as pack_a in the AVX2 unit). E converts FP16/BF16 to FP32.
 */
template <class E>
void pack_a(const typename E::T* A, std::ptrdiff_t lda, bool trans, int mc, int kc, float* Ap) noexcept {
    for (int i = 0; i < mc; i += MR) {
        const int rows = std::min(MR, mc - i);
        float* dst = Ap + static_cast<std::ptrdiff_t>(i) * kc;
        if (trans) {
            const typename E::T* src = A + i;
            for (int p = 0; p < kc; ++p)
                for (int r = 0; r < MR; ++r) dst[p * MR + r] = r < rows ? E::one(src + p * lda + r) : 0.0f;
        } else {
            const typename E::T* src = A + i * lda;
            for (int p = 0; p < kc; ++p)
                for (int r = 0; r < MR; ++r) dst[p * MR + r] = r < rows ? E::one(src + r * lda + p) : 0.0f;
        }
    }
}

/**
 * pack_b — op(B) into 32-column micro-panels, columns past nc zero; trans: B
 * stored N×K. E converts FP16/BF16 to FP32 (16 values per conversion).
 */
template <class E>
void pack_b(const typename E::T* B, std::ptrdiff_t ldb, bool trans, int kc, int nc, float* Bp) noexcept {
    for (int j = 0; j < nc; j += NR) {
        const int cols = std::min(NR, nc - j);
        float* dst = Bp + static_cast<std::ptrdiff_t>(j) * kc;
        if (trans) {  // each panel column is one contiguous row of B
            for (int c = 0; c < NR; ++c) {
                const typename E::T* src = B + (j + c) * ldb;
                for (int p = 0; p < kc; ++p) dst[p * NR + c] = c < cols ? E::one(src + p) : 0.0f;
            }
            continue;
        }
        const typename E::T* src = B + j;
        const __mmask16 m0 = static_cast<__mmask16>(cols >= 16 ? 0xFFFFu : (1u << cols) - 1u);
        const __mmask16 m1 = static_cast<__mmask16>(cols <= 16 ? 0u : cols >= 32 ? 0xFFFFu : (1u << (cols - 16)) - 1u);
        for (int p = 0; p < kc; ++p) {
            _mm512_storeu_ps(dst + p * NR, E::sixteen(m0, src + p * ldb));
            _mm512_storeu_ps(dst + p * NR + 16, E::sixteen(m1, src + p * ldb + 16));
        }
    }
}

// ─── trans_b with few rows: row-streaming GEMV (as in the AVX2 unit) ─────────
/** Lanes [0, n) of a 16-lane mask; n may be <= 0 or >= 16. */
SIMD_ML_ALWAYS_INLINE __mmask16 first_lanes(int n) noexcept {
    return static_cast<__mmask16>(n >= 16 ? 0xFFFFu : n <= 0 ? 0u : (1u << n) - 1u);
}

/**
 * dot_rows<EB, J> — C[i][j..j+J) += alpha · Σ_p a[p] · B[j+q][p], J <= 8 rows
 * of B (stored N×K, elements EB::T) and one FP32 row a: J streams of B, 32
 * values per row per step in 2 ZMM accumulators (16 for J = 8, of 32
 * registers), FP16/BF16 converted in registers (16 values per conversion), K
 * tail with masked loads. Bound by reading B, like the AVX2 version, with the
 * same row counts (4 FP32, 8 FP16/BF16; chosen on the AVX2 core); its speed
 * is measured on AVX-512 hardware only (docs/BENCHMARKS.md).
 */
template <class EB, int J>
SIMD_ML_ALWAYS_INLINE void dot_rows(int K, const float* __restrict__ a, const typename EB::T* __restrict__ B,
                                    std::ptrdiff_t ldb, float* __restrict__ c, float alpha) noexcept {
    __m512 acc[J][2];
    for (int q = 0; q < J; ++q) acc[q][0] = acc[q][1] = _mm512_setzero_ps();
    int p = 0;
    for (; p + 32 <= K; p += 32) {
        const __m512 a0 = _mm512_loadu_ps(a + p), a1 = _mm512_loadu_ps(a + p + 16);
        for (int q = 0; q < J; ++q) {
            const typename EB::T* b = B + q * ldb + p;
            acc[q][0] = _mm512_fmadd_ps(a0, EB::sixteen(0xFFFF, b), acc[q][0]);
            acc[q][1] = _mm512_fmadd_ps(a1, EB::sixteen(0xFFFF, b + 16), acc[q][1]);
        }
    }
    if (p < K) {
        const __mmask16 m0 = first_lanes(K - p), m1 = first_lanes(K - p - 16);
        const __m512 a0 = _mm512_maskz_loadu_ps(m0, a + p), a1 = _mm512_maskz_loadu_ps(m1, a + p + 16);
        for (int q = 0; q < J; ++q) {
            const typename EB::T* b = B + q * ldb + p;
            acc[q][0] = _mm512_fmadd_ps(a0, EB::sixteen(m0, b), acc[q][0]);
            acc[q][1] = _mm512_fmadd_ps(a1, EB::sixteen(m1, b + 16), acc[q][1]);
        }
    }
    for (int q = 0; q < J; ++q) c[q] += alpha * _mm512_reduce_add_ps(_mm512_add_ps(acc[q][0], acc[q][1]));
}

/** C (prescaled by beta) += alpha · A · Bᵀ, A FP32, B stored N×K (EB::T), R rows of B at a time. */
template <class EB>
void gemm_dot_trans_b(int M, int N, int K, float alpha, const float* A, std::ptrdiff_t lda,
                      const typename EB::T* B, std::ptrdiff_t ldb, float* C, std::ptrdiff_t ldc) noexcept {
    constexpr int R = sizeof(typename EB::T) == 2 ? 8 : 4;
    for (int j = 0; j < N; j += R) {
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

struct Blocking { int kc, mc, nc; };

const Blocking& blocking() noexcept {
    static const Blocking b = []() noexcept {
        const dispatch::GemmBlocking g = dispatch::gemm_blocking(MR, NR);
        return Blocking{g.kc, g.mc, g.nc};
    }();
    return b;
}

/** Per-thread packing buffers, grown on demand and kept (see the AVX2 unit). */
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

/** Operands as the loops see them: op(A) is M×K, op(B) is K×N, elements E::T. */
template <class E>
struct Operands {
    using T = typename E::T;
    const T* A; std::ptrdiff_t lda; bool ta;
    const T* B; std::ptrdiff_t ldb; bool tb;
    const T* a_at(int i, int p) const { return ta ? A + p * lda + i : A + i * lda + p; }
    const T* b_at(int p, int j) const { return tb ? B + j * ldb + p : B + p * ldb + j; }
};

template <class E>
void gemm_column_block(int M, int jc, int nc, int K, float alpha, const Operands<E>& op,
                       float* C, std::ptrdiff_t ldc, const Blocking& bl) {
    PackBuffers& buf = thread_buffers();
    const int kc_max = std::min(K, bl.kc);
    const int mc_max = std::min(M, bl.mc);
    float* Bp = buf.get_b(static_cast<std::size_t>(kc_max) * ((nc + NR - 1) / NR * NR));
    float* Ap = buf.get_a(static_cast<std::size_t>(kc_max) * ((mc_max + MR - 1) / MR * MR));
    for (int pc = 0; pc < K; pc += bl.kc) {
        const int kc = std::min(K - pc, bl.kc);
        pack_b<E>(op.b_at(pc, jc), op.ldb, op.tb, kc, nc, Bp);
        for (int ic = 0; ic < M; ic += bl.mc) {
            const int mc = std::min(M - ic, bl.mc);
            pack_a<E>(op.a_at(ic, pc), op.lda, op.ta, mc, kc, Ap);
            for (int jr = 0; jr < nc; jr += NR) {
                const int n = std::min(NR, nc - jr);
                const float* Bm = Bp + static_cast<std::ptrdiff_t>(jr) * kc;
                for (int ir = 0; ir < mc; ir += MR)
                    run_tile(std::min(MR, mc - ir), n, kc, Ap + static_cast<std::ptrdiff_t>(ir) * kc,
                             Bm, C + (ic + ir) * ldc + jc + jr, ldc, alpha);
            }
        }
    }
}

#ifdef SIMD_ML_OPENMP
/** Loops 2-5 of one column block on every thread of a team (as in the AVX2 unit). */
template <class E>
void gemm_column_block_team(int M, int jc, int nc, int K, float alpha, const Operands<E>& op,
                            float* C, std::ptrdiff_t ldc, const Blocking& bl, float* Bp,
                            float* Ap) noexcept {
    const int team = omp_get_num_threads(), t = omp_get_thread_num();
    const GemmChunks ch = gemm_chunks(team, M, nc, MR, NR, bl.mc);
    const int chunks = ch.m_chunks * ch.n_chunks;
    const int panels = (nc + NR - 1) / NR;
    const int b0 = static_cast<int>(static_cast<long long>(t) * panels / team) * NR;
    const int b1 = std::min(nc, static_cast<int>(static_cast<long long>(t + 1) * panels / team) * NR);
    for (int pc = 0; pc < K; pc += bl.kc) {
        const int kc = std::min(K - pc, bl.kc);
        if (b1 > b0) pack_b<E>(op.b_at(pc, jc + b0), op.ldb, op.tb, kc, b1 - b0, Bp + static_cast<std::ptrdiff_t>(b0) * kc);
        int packed_ic = -1;
#pragma omp barrier
#pragma omp for schedule(dynamic, 1)
        for (int c = 0; c < chunks; ++c) {
            const int ic = c / ch.n_chunks * ch.mc, j0 = c % ch.n_chunks * ch.nc;
            const int mc = std::min(M - ic, ch.mc), j1 = std::min(nc, j0 + ch.nc);
            if (ic != packed_ic) { pack_a<E>(op.a_at(ic, pc), op.lda, op.ta, mc, kc, Ap); packed_ic = ic; }
            for (int jr = j0; jr < j1; jr += NR) {
                const int n = std::min(NR, nc - jr);
                const float* Bm = Bp + static_cast<std::ptrdiff_t>(jr) * kc;
                for (int ir = 0; ir < mc; ir += MR)
                    run_tile(std::min(MR, mc - ir), n, kc, Ap + static_cast<std::ptrdiff_t>(ir) * kc,
                             Bm, C + (ic + ir) * ldc + jc + jr, ldc, alpha);
            }
        }
    }
}

/** Loops 1-5 on `threads` threads; buffers allocated before the region (as in the AVX2 unit). */
template <class E>
void gemm_packed_team(int threads, int M, int N, int K, float alpha, const Operands<E>& op,
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
        for (int jc = 0; jc < N; jc += bl.nc)
            gemm_column_block_team(M, jc, std::min(N - jc, bl.nc), K, alpha, op, C, ldc, bl, Bp, Ap_t);
    }
}
#endif

/** Loops 1-5 (C already scaled by beta, K > 0). */
template <class E>
void packed(int M, int N, int K, float alpha, const Operands<E>& op, float* C, std::ptrdiff_t ldc) {
    const Blocking& bl = blocking();
#ifdef SIMD_ML_OPENMP
    if (const int threads = gemm_threads(M, N, K, MR, NR); threads > 1) {
        gemm_packed_team(threads, M, N, K, alpha, op, C, ldc, bl);
        return;
    }
#endif
    for (int jc = 0; jc < N; jc += bl.nc)
        gemm_column_block(M, jc, std::min(N - jc, bl.nc), K, alpha, op, C, ldc, bl);
}

/** FP16/BF16 packed path: conversion in pack_a/pack_b, the 12×32 FP32 kernel inside. */
template <class E>
void sgemm_lowp_packed(bool tb, int M, int N, int K, float alpha, const typename E::T* A, int lda,
                       const typename E::T* B, int ldb, float beta, float* C, int ldc) {
    scale_matrix_c(C, M, N, ldc, beta);
    if (K == 0) return;
    packed<E>(M, N, K, alpha, Operands<E>{A, lda, false, B, ldb, tb}, C, ldc);
}

/**
 * FP16/BF16 with trans_b and at most kDotRowsMax rows (x·Wᵀ, decode): the
 * rows of A converted to FP32 once, then the row-streaming GEMV on B in place.
 */
template <class E>
void sgemm_lowp_dot(int M, int N, int K, float alpha, const typename E::T* A, int lda,
                    const typename E::T* B, int ldb, float beta, float* C, int ldc) {
    scale_matrix_c(C, M, N, ldc, beta);
    if (K == 0) return;
    float* Af = thread_buffers().get_a(static_cast<std::size_t>(M) * K);
    for (int i = 0; i < M; ++i) {
        const typename E::T* src = A + static_cast<std::ptrdiff_t>(i) * lda;
        float* dst = Af + static_cast<std::ptrdiff_t>(i) * K;
        for (int p = 0; p < K; p += 16) {
            const __mmask16 m = first_lanes(K - p);
            _mm512_mask_storeu_ps(dst + p, m, E::sixteen(m, src + p));
        }
    }
    gemm_dot_trans_b<E>(M, N, K, alpha, Af, K, B, ldb, C, ldc);
}

}  // namespace

void sgemm_blocking_avx512(int* kc, int* mc, int* nc) noexcept {
    const Blocking& b = blocking();
    *kc = b.kc; *mc = b.mc; *nc = b.nc;
}

void sgemm_packed_avx512(bool trans_a, bool trans_b, int M, int N, int K,
                         float alpha,
                         const float* A, int lda,
                         const float* B, int ldb,
                         float beta,
                         float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    // x·Wᵀ with a few rows: this unit's row-streaming GEMV. Other small
    // products: the AVX2 direct path (in the AVX2 unit).
    if (trans_b && !trans_a && M <= kDotRowsMax) {
        scale_matrix_c(C, M, N, ldc, beta);
        if (K > 0) gemm_dot_trans_b<F32>(M, N, K, alpha, A, lda, B, ldb, C, ldc);
        return;
    }
    if (!trans_b && sgemm_avx2_prefers_direct(M, N, K)) {
        sgemm_direct_avx2(trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    scale_matrix_c(C, M, N, ldc, beta);
    if (K == 0) return;
    packed<F32>(M, N, K, alpha, Operands<F32>{A, lda, trans_a, B, ldb, trans_b}, C, ldc);
}

// x·Wᵀ with a few rows: the row-streaming GEMV; other small products: the AVX2
// direct path, as for FP32.
void sgemm_f16_avx512(bool trans_b, int M, int N, int K, float alpha, const std::uint16_t* A, int lda,
                      const std::uint16_t* B, int ldb, float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    if (trans_b && M <= kDotRowsMax)
        sgemm_lowp_dot<F16>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    else if (!trans_b && sgemm_avx2_prefers_direct(M, N, K))
        sgemm_f16_avx2(false, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        sgemm_lowp_packed<F16>(trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// INT8: AVX512_VNNI (avx512vnni_gemm.cpp) where the CPU has it, else the AVX2
// table's choice (AVX-VNNI, or the FP32 conversion path).
void sgemm_u8s8_avx512(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                       int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                       const float* b_scales, float* C, int ldc) {
#ifdef SIMD_ML_HAVE_AVX512VNNI
    if (avx512vnni_available()) {
        sgemm_u8s8_avx512vnni(trans_b, M, N, K, A, lda, a_zero_point, B, ldb, a_scale, b_scales, C, ldc);
        return;
    }
#endif
    sgemm_u8s8_avx2(trans_b, M, N, K, A, lda, a_zero_point, B, ldb, a_scale, b_scales, C, ldc);
}

// Packed shapes convert at pack time and run the FP32 kernel. vdpbf16ps
// (avx512bf16_gemm.cpp) is opt-in, SIMD_ML_BF16_DOT=1 with AVX512_BF16 and one
// thread: on Emerald Rapids it ran at 64-73% of this path (DESIGN.md §4.1).
// The vdpbf16ps path has no trans_b; x·Wᵀ goes to the GEMV or packing.
void sgemm_bf16_avx512(bool trans_b, int M, int N, int K, float alpha, const std::uint16_t* A, int lda,
                       const std::uint16_t* B, int ldb, float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    if (trans_b && M <= kDotRowsMax) {
        sgemm_lowp_dot<BF16>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    if (!trans_b && sgemm_avx2_prefers_direct(M, N, K)) {
        sgemm_bf16_avx2(false, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#ifdef SIMD_ML_HAVE_AVX512BF16
    if (!trans_b && bf16_dot_enabled() && gemm_threads(M, N, K, MR, NR) == 1) {
        sgemm_bf16_avx512bf16(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif
    sgemm_lowp_packed<BF16>(trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // namespace gemm
}  // namespace simd_ml
