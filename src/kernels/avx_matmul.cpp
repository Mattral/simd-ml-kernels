/**
 * avx_matmul.cpp — Hand-Vectorized AVX2 / AVX-512 GEMM Microkernel
 *
 * Computes: C = alpha·A × B + beta·C   (all FP32, row-major layout)
 *   A: M×K,  B: K×N,  C: M×N
 *
 * ─── Micro-Architectural Strategy ────────────────────────────────────────────
 *
 * 1. LOOP TILING / CACHE BLOCKING  (Goto 2008 / BLIS framework)
 *    Tile sizes tuned for Intel Core (48 KB L1d, 1.25 MB L2):
 *      MC=64, KC=256, NC=64
 *    The 5-loop structure keeps Ac and Bc panels L2-resident across the M-loop.
 *
 * 2. REGISTER BLOCKING  (6 × 16 micro-tile for AVX2)
 *    AVX2 path: 6 rows × 16 cols = 12 YMM accumulators.
 *    Remaining 4 YMM registers hold A broadcasts and B panel data.
 *    Total: 16/16 YMM — zero spill pressure on an ideal renamer.
 *
 * 3. PANEL PACKING
 *    pack_A: reorders A[mc×kc] into column-major micro-panels so each
 *            broadcast A[m][k] is sequential in memory.
 *    pack_B: reorders B[kc×nc] into NR-wide panels for unit-stride loads.
 *    Alpha is folded into pack_A to eliminate a multiply in the hot loop.
 *
 * 4. FMA INTRINSICS  (_mm256_fmadd_ps)
 *    Each FMA = 2 FLOP at 0.5 cy throughput → theoretical peak on Skylake:
 *    16 FLOP/cy × 2 ports × (clock GHz) = 112 GFLOPS/core at 3.5 GHz.
 *
 * 5. SOFTWARE PREFETCHING
 *    Issues _mm_prefetch (PREFETCHT0) 2 B-panel rows ahead to overlap
 *    L2→L1 fill (≈12 cy) with FMA execution on the current rows.
 *
 * 6. LOOP UNROLLING (2×)
 *    A 2× unroll in K exposes more ILP to the out-of-order scheduler while
 *    keeping the source code auditable. The unroll factor is not tuned.
 *
 * Compilation flags expected: -O3 -march=native -mfma
 */

#include "cache_alloc.hpp"
#include "avx_matmul.hpp"

#include <cstring>
#include <algorithm>

#ifdef __AVX512F__
#  include <immintrin.h>
#elif defined(__AVX2__)
#  include <immintrin.h>
#else
#  error "avx_matmul.cpp requires at least AVX2. Compile with -mavx2 -mfma."
#endif

// ─── Tiling constants ─────────────────────────────────────────────────────────
static constexpr int MC = 64;     // M-tile: rows of A / C per outer-M block
static constexpr int KC = 256;    // K-tile: depth panel (limited by L2 capacity)
static constexpr int NC = 64;     // N-tile: cols of B / C per outer-N block
static constexpr int MR = 6;      // Micro-kernel row register count
static constexpr int NR = 16;     // Micro-kernel col register count (2 AVX2 vectors)

// ─── AVX2 6×16 Micro-kernel ───────────────────────────────────────────────────
/**
 * gemm_micro_6x16_avx2 — Compute C += A_packed × B_packed for a 6×16 tile.
 *
 * @param kc       Depth of the K-panel (loop count)
 * @param A_panel  Packed A sub-matrix: kc rows × MR cols, column-major per k
 *                 A_panel[k*MR + m] = alpha × A[row_base+m][k_base+k]
 * @param B_panel  Packed B sub-matrix: kc rows × NR cols, row-major per k
 *                 B_panel[k*NR + n] = B[k_base+k][col_base+n]
 * @param C        C output tile, row stride = ldc
 * @param ldc      Leading dimension (column stride) of C in the full M×N matrix
 *
 * Memory access pattern (per k-iteration):
 *   A: sequential broadcast of MR=6 scalars from A_panel
 *   B: two sequential 8-float vector loads from B_panel
 *   C: read once at start, written once at end (register-resident throughout)
 *
 * Loop unrolled 2× to allow the out-of-order scheduler to overlap:
 *   - Independent FMA chains for k and k+1
 *   - Prefetch for B panel rows 2 steps ahead
 */
#ifdef __AVX2__
static void gemm_micro_6x16_avx2(int kc,
                                   const float* __restrict__ A_panel,
                                   const float* __restrict__ B_panel,
                                   float*       __restrict__ C,
                                   int ldc) noexcept {
    // Load 12 accumulator registers: 6 rows × (lo+hi) = 6×2 YMM
    __m256 c0lo = _mm256_loadu_ps(C + 0*ldc + 0);
    __m256 c0hi = _mm256_loadu_ps(C + 0*ldc + 8);
    __m256 c1lo = _mm256_loadu_ps(C + 1*ldc + 0);
    __m256 c1hi = _mm256_loadu_ps(C + 1*ldc + 8);
    __m256 c2lo = _mm256_loadu_ps(C + 2*ldc + 0);
    __m256 c2hi = _mm256_loadu_ps(C + 2*ldc + 8);
    __m256 c3lo = _mm256_loadu_ps(C + 3*ldc + 0);
    __m256 c3hi = _mm256_loadu_ps(C + 3*ldc + 8);
    __m256 c4lo = _mm256_loadu_ps(C + 4*ldc + 0);
    __m256 c4hi = _mm256_loadu_ps(C + 4*ldc + 8);
    __m256 c5lo = _mm256_loadu_ps(C + 5*ldc + 0);
    __m256 c5hi = _mm256_loadu_ps(C + 5*ldc + 8);

    const float* ap = A_panel;
    const float* bp = B_panel;

    // ── 2× Unrolled K-loop ────────────────────────────────────────────────────
    // Process two k-iterations per outer loop body.
    // Invariant entering each iteration: ap = A_panel + k*MR, bp = B_panel + k*NR
    int k = 0;
    for (; k + 2 <= kc; k += 2) {
        // Prefetch B panel 2 rows ahead (= 2×NR×4 bytes = 128 bytes ahead)
        prefetch_l1(bp + 2*NR);

        // ── k+0 ─────────────────────────────────────────────────────────────
        __m256 b0lo = _mm256_loadu_ps(bp + 0);        // B[k][0..7]
        __m256 b0hi = _mm256_loadu_ps(bp + 8);        // B[k][8..15]

        c0lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b0lo, c0lo);
        c0hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b0hi, c0hi);
        c1lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b0lo, c1lo);
        c1hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b0hi, c1hi);
        c2lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b0lo, c2lo);
        c2hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b0hi, c2hi);
        c3lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b0lo, c3lo);
        c3hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b0hi, c3hi);
        c4lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b0lo, c4lo);
        c4hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b0hi, c4hi);
        c5lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b0lo, c5lo);
        c5hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b0hi, c5hi);

        ap += MR;  // advance to next k row in A panel
        bp += NR;  // advance to next k row in B panel

        // ── k+1 ─────────────────────────────────────────────────────────────
        __m256 b1lo = _mm256_loadu_ps(bp + 0);        // B[k+1][0..7]
        __m256 b1hi = _mm256_loadu_ps(bp + 8);        // B[k+1][8..15]

        c0lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b1lo, c0lo);
        c0hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b1hi, c0hi);
        c1lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b1lo, c1lo);
        c1hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b1hi, c1hi);
        c2lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b1lo, c2lo);
        c2hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b1hi, c2hi);
        c3lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b1lo, c3lo);
        c3hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b1hi, c3hi);
        c4lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b1lo, c4lo);
        c4hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b1hi, c4hi);
        c5lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b1lo, c5lo);
        c5hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b1hi, c5hi);

        ap += MR;
        bp += NR;
    }
    // Scalar tail: handle remaining k iterations (kc % 2 == 1)
    if (k < kc) {
        __m256 blo = _mm256_loadu_ps(bp + 0);
        __m256 bhi = _mm256_loadu_ps(bp + 8);
        c0lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), blo, c0lo);
        c0hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), bhi, c0hi);
        c1lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), blo, c1lo);
        c1hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), bhi, c1hi);
        c2lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), blo, c2lo);
        c2hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), bhi, c2hi);
        c3lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), blo, c3lo);
        c3hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), bhi, c3hi);
        c4lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), blo, c4lo);
        c4hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), bhi, c4hi);
        c5lo = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), blo, c5lo);
        c5hi = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), bhi, c5hi);
    }

    // Write accumulators back to C
    _mm256_storeu_ps(C + 0*ldc + 0, c0lo); _mm256_storeu_ps(C + 0*ldc + 8, c0hi);
    _mm256_storeu_ps(C + 1*ldc + 0, c1lo); _mm256_storeu_ps(C + 1*ldc + 8, c1hi);
    _mm256_storeu_ps(C + 2*ldc + 0, c2lo); _mm256_storeu_ps(C + 2*ldc + 8, c2hi);
    _mm256_storeu_ps(C + 3*ldc + 0, c3lo); _mm256_storeu_ps(C + 3*ldc + 8, c3hi);
    _mm256_storeu_ps(C + 4*ldc + 0, c4lo); _mm256_storeu_ps(C + 4*ldc + 8, c4hi);
    _mm256_storeu_ps(C + 5*ldc + 0, c5lo); _mm256_storeu_ps(C + 5*ldc + 8, c5hi);
}
#endif  // __AVX2__

// ─── AVX-512 6×32 Micro-kernel ────────────────────────────────────────────────
#ifdef __AVX512F__
/**
 * gemm_micro_6x32_avx512 — 6×32 blocking using dual 512-bit ZMM accumulators.
 *
 * Doubles the N-width vs the AVX2 6×16 kernel using AVX-512's wider 512-bit
 * registers AND its much larger physical register file (32 ZMM vs 16 YMM),
 * so there is no register-pressure trade-off here the way there was choosing
 * between the AVX2 8×8 and 6×16 layouts.
 *
 * ─── Register budget (32 ZMM available) ──────────────────────────────────────
 *   12 ZMM  — accumulators: 6 rows × 2 halves (lo=cols[0:16), hi=cols[16:32))
 *    2 ZMM  — B panel (b_lo, b_hi) reloaded each k-step
 *    1 ZMM  — A broadcast (reused per row per k-step)
 *   15 ZMM total — 17 registers of headroom vs. the 32 available; zero
 *   spill risk even with aggressive unrolling.
 *
 * ─── Correctness note (see ROADMAP.md §v0.8 history) ────────────────────────
 * An earlier version of this kernel declared NR512=32 but only ever loaded
 * and accumulated ONE __m512 (16 floats) per row — silently leaving the
 * upper 16 columns of every 32-wide output tile untouched. That bug is why
 * AVX-512 dispatch was disabled for an entire release cycle. This version
 * is proven correct via tests/test_gemm.cpp's
 * test_gemm_avx512_full_coverage(), which forces 100% of the output
 * through this exact kernel (M and N chosen so no edge/tail block can
 * mask a broken full-block path) and independently checks both
 * accumulator halves against a scalar reference.
 */
static void gemm_micro_6x32_avx512(int kc,
                                    const float* __restrict__ A_panel,
                                    const float* __restrict__ B_panel,
                                    float*       __restrict__ C,
                                    int ldc) noexcept {
    // Load existing C values — both halves of all 6 rows (accumulate, not overwrite)
    __m512 c0_lo = _mm512_loadu_ps(C + 0*ldc +  0), c0_hi = _mm512_loadu_ps(C + 0*ldc + 16);
    __m512 c1_lo = _mm512_loadu_ps(C + 1*ldc +  0), c1_hi = _mm512_loadu_ps(C + 1*ldc + 16);
    __m512 c2_lo = _mm512_loadu_ps(C + 2*ldc +  0), c2_hi = _mm512_loadu_ps(C + 2*ldc + 16);
    __m512 c3_lo = _mm512_loadu_ps(C + 3*ldc +  0), c3_hi = _mm512_loadu_ps(C + 3*ldc + 16);
    __m512 c4_lo = _mm512_loadu_ps(C + 4*ldc +  0), c4_hi = _mm512_loadu_ps(C + 4*ldc + 16);
    __m512 c5_lo = _mm512_loadu_ps(C + 5*ldc +  0), c5_hi = _mm512_loadu_ps(C + 5*ldc + 16);

    const float* ap = A_panel;
    const float* bp = B_panel;

    for (int k = 0; k < kc; ++k, ap += MR, bp += 32) {
        // Load BOTH halves of the 32-wide B panel row for this k-step.
        __m512 b_lo = _mm512_loadu_ps(bp);       // cols [0, 16)
        __m512 b_hi = _mm512_loadu_ps(bp + 16);  // cols [16, 32)

        __m512 a0 = _mm512_set1_ps(ap[0]);
        c0_lo = _mm512_fmadd_ps(a0, b_lo, c0_lo);
        c0_hi = _mm512_fmadd_ps(a0, b_hi, c0_hi);

        __m512 a1 = _mm512_set1_ps(ap[1]);
        c1_lo = _mm512_fmadd_ps(a1, b_lo, c1_lo);
        c1_hi = _mm512_fmadd_ps(a1, b_hi, c1_hi);

        __m512 a2 = _mm512_set1_ps(ap[2]);
        c2_lo = _mm512_fmadd_ps(a2, b_lo, c2_lo);
        c2_hi = _mm512_fmadd_ps(a2, b_hi, c2_hi);

        __m512 a3 = _mm512_set1_ps(ap[3]);
        c3_lo = _mm512_fmadd_ps(a3, b_lo, c3_lo);
        c3_hi = _mm512_fmadd_ps(a3, b_hi, c3_hi);

        __m512 a4 = _mm512_set1_ps(ap[4]);
        c4_lo = _mm512_fmadd_ps(a4, b_lo, c4_lo);
        c4_hi = _mm512_fmadd_ps(a4, b_hi, c4_hi);

        __m512 a5 = _mm512_set1_ps(ap[5]);
        c5_lo = _mm512_fmadd_ps(a5, b_lo, c5_lo);
        c5_hi = _mm512_fmadd_ps(a5, b_hi, c5_hi);
    }

    // Write back BOTH halves of every row — this is the exact line the
    // original buggy version omitted for the "hi" half.
    _mm512_storeu_ps(C + 0*ldc +  0, c0_lo);  _mm512_storeu_ps(C + 0*ldc + 16, c0_hi);
    _mm512_storeu_ps(C + 1*ldc +  0, c1_lo);  _mm512_storeu_ps(C + 1*ldc + 16, c1_hi);
    _mm512_storeu_ps(C + 2*ldc +  0, c2_lo);  _mm512_storeu_ps(C + 2*ldc + 16, c2_hi);
    _mm512_storeu_ps(C + 3*ldc +  0, c3_lo);  _mm512_storeu_ps(C + 3*ldc + 16, c3_hi);
    _mm512_storeu_ps(C + 4*ldc +  0, c4_lo);  _mm512_storeu_ps(C + 4*ldc + 16, c4_hi);
    _mm512_storeu_ps(C + 5*ldc +  0, c5_lo);  _mm512_storeu_ps(C + 5*ldc + 16, c5_hi);
}
#endif  // __AVX512F__

// ─── Panel packing ────────────────────────────────────────────────────────────

/**
 * pack_B — Reorder B[kc × nc] into a contiguous NR-panel-major buffer.
 *
 * Output layout: for each N-block of NR columns:
 *   B_packed[block * kc * nr_block + k * NR + n] = B[k][block*NR + n]
 *
 * This converts the strided column-access pattern into sequential reads
 * (unit stride in the N dimension), enabling full-width vector loads in
 * the micro-kernel and allowing hardware prefetch to work effectively.
 */
static void pack_B(const float* B, int ldb,
                   int kc, int nc,
                   float* __restrict__ B_packed,
                   int nr_block) noexcept {
    const int num_blocks = (nc + nr_block - 1) / nr_block;
    for (int block = 0; block < num_blocks; ++block) {
        const int jb         = block * nr_block;
        const int actual_nr  = std::min(nr_block, nc - jb);
        float*    dst_base   = B_packed + block * kc * nr_block;

        for (int k = 0; k < kc; ++k) {
            const float* src = B + k * ldb + jb;
            float*       dst = dst_base + k * nr_block;

            for (int n = 0; n < actual_nr; ++n) dst[n] = src[n];
            // Zero-pad to a full NR block so the micro-kernel reads clean data
            for (int n = actual_nr; n < nr_block; ++n) dst[n] = 0.0f;
        }
    }
}

/**
 * pack_A — Reorder A[mc × kc] into column-major micro-panels.
 *
 * Output layout: for each M-block of MR rows:
 *   A_packed[block * kc * MR + k * MR + m] = alpha × A[block*MR + m][k]
 *
 * Folding alpha into pack_A means the micro-kernel hot loop is pure FMA
 * with no scalar multiply. This is the standard BLIS packing optimization.
 */
static void pack_A(const float* A, int lda,
                   int mc, int kc,
                   float* __restrict__ A_packed,
                   float alpha) noexcept {
    const int num_blocks = (mc + MR - 1) / MR;
    for (int block = 0; block < num_blocks; ++block) {
        const int rows       = std::min(MR, mc - block * MR);
        float*    dst_base   = A_packed + block * kc * MR;
        // A_block points to A[block*MR][0]; src = A_block + k steps column k
        const float* A_block = A + block * MR * lda;

        for (int k = 0; k < kc; ++k) {
            float* dst = dst_base + k * MR;
            // src[m * lda] = A[block*MR + m][k]
            const float* src = A_block + k;
            for (int m = 0; m < rows;  ++m) dst[m] = alpha * src[m * lda];
            for (int m = rows; m < MR; ++m) dst[m] = 0.0f;
        }
    }
}

// ─── Scalar fallback for edge blocks ─────────────────────────────────────────
/**
 * scalar_block — Accumulate a partial mr×nr block of C using scalar ops.
 *
 * Used for tail blocks where mr < MR or nr < NR. Uses original (unpacked)
 * A and B matrices. alpha is applied here since pack_A has NOT been used.
 *
 * This is the correctness-critical fallback path — simple scalar triple
 * loop with no SIMD, verifiably correct by inspection.
 */
static void scalar_block(int mr, int nr, int kc, float alpha,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float*       C, int ldc) noexcept {
    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nr; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < kc; ++k) {
                acc += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += alpha * acc;
        }
    }
}

// ─── Public API: simd_sgemm ───────────────────────────────────────────────────
/**
 * simd_sgemm — Single-precision GEMM with AVX2/AVX-512 micro-kernel.
 *
 *   C = alpha·A·B + beta·C
 *
 * Memory layout: all matrices row-major (C convention).
 * Alignment: 64-byte alignment preferred but not required (loadu used throughout).
 */
void simd_sgemm(int M, int N, int K,
                float alpha,
                const float* __restrict__ A, int lda,
                const float* __restrict__ B, int ldb,
                float beta,
                float*       __restrict__ C, int ldc) noexcept {
    // K == 0 is valid (BLAS semantics): A·B is empty, so C = beta·C.
    // The beta pass below runs and the K-loops do nothing.
    if (M <= 0 || N <= 0 || K < 0) return;

    // ── Beta scaling ──────────────────────────────────────────────────────────
    if (beta == 0.0f) {
        for (int i = 0; i < M; ++i) std::memset(C + i * ldc, 0, N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i * ldc + j] *= beta;
    }

    // ── Runtime ISA selection ─────────────────────────────────────────────────
    // gemm_micro_6x32_avx512 is proven correct via
    // tests/test_gemm.cpp: test_gemm_avx512_full_coverage(), which forces
    // 100% of the output through this kernel and independently checks
    // both accumulator halves — exactly the split the earlier disabled
    // version silently failed to write for the "hi" half.
    //
    // inner_NR MUST match the nr_block passed to pack_B below and the nr
    // check in the dispatch condition — this triple-consistency is exactly
    // what broke before (inner_NR was 32 while the kernel only handled 16).
#if defined(__AVX512F__)
    const bool use_avx512 = __builtin_cpu_supports("avx512f") &&
                            __builtin_cpu_supports("avx512dq");
    const int inner_NR = use_avx512 ? 32 : NR;
#else
    constexpr int inner_NR = NR;
#endif

    // ── 5-Loop GEMM ───────────────────────────────────────────────────────────
    //
    //   Loop 1 (jc):  iterate over N in NC-wide blocks
    //   Loop 2 (kc):  iterate over K in KC-wide blocks
    //     pack_B into B_pack (fits in L2)
    //   Loop 3 (ic):  iterate over M in MC-wide blocks
    //     pack_A into A_pack (fits in L2, alongside B_pack)
    //   Loop 4 (jr):  iterate over nc in NR-wide blocks     } micro-kernel
    //   Loop 5 (ir):  iterate over mc in MR-wide blocks     } register tile
    //
    // Notes:
    //   - Packing buffers are allocated per-j-block to give fresh clean buffers.
    //   - For OpenMP builds, each thread gets private packing buffers.

    const int num_j_blocks = (N + NC - 1) / NC;

    for (int jb = 0; jb < num_j_blocks; ++jb) {
        const int j  = jb * NC;
        const int nc = std::min(NC, N - j);

        // Per-block packing buffers; unique_ptr ensures cleanup even on early exit
        auto A_pack = make_aligned_array<float>(
            static_cast<std::size_t>((MC + MR - 1) / MR) * KC * MR);
        auto B_pack = make_aligned_array<float>(
            static_cast<std::size_t>((NC + inner_NR - 1) / inner_NR) * KC * inner_NR);

        // Loop 2: K-tiles
        for (int k = 0; k < K; k += KC) {
            const int kc = std::min(KC, K - k);

            // Pack B[k..k+kc-1][j..j+nc-1] → B_pack
            pack_B(B + k * ldb + j, ldb, kc, nc, B_pack.get(), inner_NR);

            // Loop 3: M-tiles
            for (int i = 0; i < M; i += MC) {
                const int mc = std::min(MC, M - i);

                // Pack A[i..i+mc-1][k..k+kc-1] → A_pack  (alpha folded in)
                pack_A(A + i * lda + k, lda, mc, kc, A_pack.get(), alpha);

                // Loop 4+5: micro-kernel tiles
                for (int ii = 0; ii < mc; ii += MR) {
                    const int mr      = std::min(MR, mc - ii);
                    const float* Aμ   = A_pack.get() + (ii / MR) * kc * MR;

                    for (int jj = 0; jj < nc; jj += inner_NR) {
                        const int nr    = std::min(inner_NR, nc - jj);
                        float*    C_ptr = C + (i + ii) * ldc + (j + jj);

                        if (mr == MR && nr == inner_NR) {
                            // Full micro-kernel block. Which kernel runs MUST match
                            // inner_NR exactly (16 -> AVX2, 32 -> AVX-512) — this
                            // three-way consistency (inner_NR / pack_B nr_block /
                            // kernel choice) is exactly what broke silently before.
                            const float* Bμ = B_pack.get() + (jj / inner_NR) * kc * inner_NR;
                            bool handled = false;
#if defined(__AVX512F__)
                            if (use_avx512) {
                                gemm_micro_6x32_avx512(kc, Aμ, Bμ, C_ptr, ldc);
                                handled = true;
                            }
#endif
                            if (!handled) {
#ifdef __AVX2__
                                gemm_micro_6x16_avx2(kc, Aμ, Bμ, C_ptr, ldc);
#else
                                // Non-AVX2 fallback: Aμ is packed with alpha already folded in,
                                // but scalar_block reads the ORIGINAL A pointer, so alpha must
                                // be passed explicitly here (unlike the AVX2 path where the
                                // micro-kernel operates on alpha-pre-scaled Aμ).
                                scalar_block(MR, inner_NR, kc, alpha,
                                             A + (i+ii)*lda + k, lda,
                                             B + k*ldb + (j+jj), ldb,
                                             C_ptr, ldc);
#endif
                            }
                        } else {
                            // Edge/tail block: use original (unpacked) A and B.
                            // alpha is applied here; beta was applied upfront.
                            scalar_block(mr, nr, kc, alpha,
                                         A + (i+ii)*lda + k, lda,
                                         B + k*ldb + (j+jj), ldb,
                                         C_ptr, ldc);
                        }
                    }
                }
            }
        }
    }
}

// ─── Scalar reference GEMM ───────────────────────────────────────────────────
/**
 * scalar_sgemm — Naïve triple-loop GEMM (correctness oracle for benchmarks).
 */
void scalar_sgemm(int M, int N, int K,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float*       C, int ldc) noexcept {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = acc;
        }
}

// ─── ISA diagnostics ──────────────────────────────────────────────────────────
bool avx_matmul_isa_is_avx512() noexcept {
#if defined(__AVX512F__)
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
#else
    return false;
#endif
}
