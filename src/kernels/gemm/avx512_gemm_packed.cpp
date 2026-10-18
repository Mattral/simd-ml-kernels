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
void sgemm_lowp_packed(int M, int N, int K, float alpha, const typename E::T* A, int lda,
                       const typename E::T* B, int ldb, float beta, float* C, int ldc) {
    scale_matrix_c(C, M, N, ldc, beta);
    if (K == 0) return;
    packed<E>(M, N, K, alpha, Operands<E>{A, lda, false, B, ldb, false}, C, ldc);
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
    // Same unpacked paths as the AVX2 table (they are in the AVX2 unit).
    const bool unpacked = trans_b ? (!trans_a && M <= 16) : sgemm_avx2_prefers_direct(M, N, K);
    if (unpacked) {
        sgemm_direct_avx2(trans_a, trans_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    scale_matrix_c(C, M, N, ldc, beta);
    if (K == 0) return;
    packed<F32>(M, N, K, alpha, Operands<F32>{A, lda, trans_a, B, ldb, trans_b}, C, ldc);
}

// Small products take the AVX2 direct path, as for FP32.
void sgemm_f16_avx512(int M, int N, int K, float alpha, const std::uint16_t* A, int lda,
                      const std::uint16_t* B, int ldb, float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    if (sgemm_avx2_prefers_direct(M, N, K))
        sgemm_f16_avx2(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        sgemm_lowp_packed<F16>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

// With AVX512_BF16 (CPUID, checked once by dispatch.cpp) and one thread,
// packed shapes use vdpbf16ps (avx512bf16_gemm.cpp); speed not measured.
void sgemm_bf16_avx512(int M, int N, int K, float alpha, const std::uint16_t* A, int lda,
                       const std::uint16_t* B, int ldb, float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    if (sgemm_avx2_prefers_direct(M, N, K)) {
        sgemm_bf16_avx2(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#ifdef SIMD_ML_HAVE_AVX512BF16
    if (bf16_avx512bf16_available() && gemm_threads(M, N, K, MR, NR) == 1) {
        sgemm_bf16_avx512bf16(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif
    sgemm_lowp_packed<BF16>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

}  // namespace gemm
}  // namespace simd_ml
