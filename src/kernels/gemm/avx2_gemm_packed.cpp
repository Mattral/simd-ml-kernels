/**
 * avx2_gemm_packed.cpp — Packed AVX2 GEMM (Goto/BLIS 5-loop structure)
 *
 * This file implements the primary GEMM used by the Python dispatch layer.
 * It uses an 8×8 register-blocked micro-kernel with explicit FMA intrinsics,
 * Goto/BLIS-style 5-loop tiling, and panel packing for cache efficiency.
 *
 * ─── Design overview ──────────────────────────────────────────────────────────
 * See docs/DESIGN.md §2–3 for cache-blocking rationale and §3.2 for the 8×8
 * register-tile choice.  Tile sizes:
 *
 *   MR=8, NR=8   — micro-kernel register block (8 YMM accumulators)
 *   KC=256        — K-panel depth: Ac+Bc fit together in L2
 *   MC=128        — M-panel height: Ac fits in L2 while B remains L3-resident
 *   NC=2048       — N-panel width: large to amortise B packing cost over M-tiles
 *
 * ─── Memory allocation strategy ──────────────────────────────────────────────
 * A_packed and B_packed are allocated ONCE per sgemm_packed call (outside all
 * loops) and reused across all k-panel and M-tile iterations.  Earlier versions
 * allocated inside the k-loop, causing O(K/KC × N/NC) heap calls per GEMM.
 *
 * ─── Thread safety (OpenMP path) ─────────────────────────────────────────────
 * When SIMD_ML_OPENMP is defined, each thread owns its own packing buffers via
 * thread_local storage, avoiding false sharing across the outer jc-parallel loop.
 */

#include "avx2_gemm_packed.hpp"
#include "cache_alloc.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <string_view>

#ifdef SIMD_ML_OPENMP
#  include <omp.h>
#endif

#ifdef __AVX2__
#  include <immintrin.h>
#else
#  error "Packed GEMM requires AVX2 support (-mavx2)"
#endif

namespace simd_ml {
namespace gemm {

// ─── Beta scaling ─────────────────────────────────────────────────────────────
static void scale_matrix_c(float* C, int M, int N, int ldc, float beta) noexcept {
    if (beta == 1.0f) return;
    for (int i = 0; i < M; ++i) {
        float* row = C + i * ldc;
        if (beta == 0.0f) {
            std::fill(row, row + N, 0.0f);
        } else {
            for (int j = 0; j < N; ++j) row[j] *= beta;
        }
    }
}

// ─── Panel packing ────────────────────────────────────────────────────────────
/**
 * pack_b_panel — Reorder B[k × n] into NR-column-major panel layout.
 *
 * Output layout: B_packed[block * k * NR + p * NR + j] = B[p][block*NR + j]
 *
 * This turns the strided column access of B into sequential streaming reads
 * in the inner kernel, enabling prefetch-friendly unit-stride loads.
 */
void pack_b_panel(const float* B, int ldb, int k, int n, float* B_packed) {
    for (int jc = 0; jc < n; jc += NR) {
        int nr = std::min(NR, n - jc);
        float* out_block = B_packed + (jc / NR) * k * NR;
        for (int p = 0; p < k; ++p) {
            const float* src = B + p * ldb + jc;
            float* dst = out_block + p * NR;
            if (nr == NR) {
                _mm256_storeu_ps(dst, _mm256_loadu_ps(src));
            } else {
                for (int j = 0; j < nr; ++j) dst[j] = src[j];
                for (int j = nr; j < NR; ++j) dst[j] = 0.0f;
            }
        }
    }
}

/**
 * pack_b_panel_avx512 — AVX-512 counterpart of pack_b_panel, blocking on
 * NR512=16 instead of NR=8. Fully independent implementation (not
 * parameterized sharing) to keep the proven AVX2 path completely untouched.
 */
#ifdef __AVX512F__
void pack_b_panel_avx512(const float* B, int ldb, int k, int n, float* B_packed) {
    for (int jc = 0; jc < n; jc += NR512) {
        int nr = std::min(NR512, n - jc);
        float* out_block = B_packed + (jc / NR512) * k * NR512;
        for (int p = 0; p < k; ++p) {
            const float* src = B + p * ldb + jc;
            float* dst = out_block + p * NR512;
            if (nr == NR512) {
                _mm512_storeu_ps(dst, _mm512_loadu_ps(src));
            } else {
                for (int j = 0; j < nr; ++j) dst[j] = src[j];
                for (int j = nr; j < NR512; ++j) dst[j] = 0.0f;
            }
        }
    }
}
#endif  // __AVX512F__

/**
 * pack_a_panel — Reorder A[m × k] into row-major micro-panel layout.
 *
 * Output layout: A_packed[i * k + p] = A[i][p]
 *
 * Each MR-row micro-tile of A becomes contiguous in memory, enabling
 * sequential scalar loads of A[i][p] during the inner kernel's k-loop.
 */
void pack_a_panel(const float* A, int lda, int m, int k, float* A_packed) {
    for (int i = 0; i < m; ++i) {
        const float* src = A + i * lda;
        float* dst = A_packed + i * k;
        std::memcpy(dst, src, sizeof(float) * static_cast<std::size_t>(k));
    }
}

// ─── Scalar edge-block fallback ───────────────────────────────────────────────
/**
 * gemm_scalar_block — Accumulate a partial mr×nr block of C.
 * Used for tail blocks where mr < MR or nr < NR.
 * Iterates in p-major order so the FP addition sequence matches the reference.
 */
static void gemm_scalar_block(int mr, int nr, int kc,
                               const float* __restrict__ A_packed,
                               const float* __restrict__ B_packed,
                               float*       __restrict__ C, int ldc,
                               float alpha) noexcept {
    for (int i = 0; i < mr; ++i) {
        float* c_row = C + i * ldc;
        const float* a_row = A_packed + i * kc;
        for (int p = 0; p < kc; ++p) {
            float a_val = a_row[p];
            const float* b_col = B_packed + p * NR;
            for (int j = 0; j < nr; ++j) {
                c_row[j] += alpha * (a_val * b_col[j]);
            }
        }
    }
}

/**
 * gemm_scalar_block_avx512 — AVX-512-path counterpart of gemm_scalar_block.
 * Identical logic, but reads the B_packed stride as NR512 (16) rather than
 * NR (8), since B_packed for this path was produced by pack_b_panel_avx512.
 * Deliberately duplicated rather than parameterized — this function is only
 * ever hit for small edge/tail blocks, so the tiny code duplication costs
 * nothing at runtime and keeps both ISA paths fully independent to verify.
 */
static void gemm_scalar_block_avx512(int mr, int nr, int kc,
                                     const float* __restrict__ A_packed,
                                     const float* __restrict__ B_packed,
                                     float*       __restrict__ C, int ldc,
                                     float alpha) noexcept {
    for (int i = 0; i < mr; ++i) {
        float* c_row = C + i * ldc;
        const float* a_row = A_packed + i * kc;
        for (int p = 0; p < kc; ++p) {
            float a_val = a_row[p];
            const float* b_col = B_packed + p * NR512;
            for (int j = 0; j < nr; ++j) {
                c_row[j] += alpha * (a_val * b_col[j]);
            }
        }
    }
}

// ─── 8×8 AVX2 Micro-kernel ───────────────────────────────────────────────────
/**
 * inner_kernel_8x8 — Compute C[0..7][0..7] += alpha * A[0..7][0..kc-1] × B[0..kc-1][0..7]
 *
 * Register allocation:
 *   acc0..acc7  — 8 YMM accumulator registers (8 rows × 1 YMM each)
 *   c0..c7      — 8 YMM registers for reading existing C values
 *   b           — 1 YMM for B panel column
 *
 * Pattern: for each k, broadcast one scalar from each A row and FMA against B.
 * This gives 8 FMAs per k-step (8 rows × 1 column), utilizing 2 FMA ports.
 *
 * Software prefetching: pull B panel 4 rows ahead to hide L2→L1 latency
 * (Skylake L2 fill ≈ 12 cycles; at 0.5 cy/FMA × 8 FMAs = 4 cycles/step).
 */
void inner_kernel_8x8(const float* __restrict__ A_packed,
                       const float* __restrict__ B_packed,
                       float*       __restrict__ C, int ldc,
                       int k_rem, float alpha) noexcept {
    // Load existing C values
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
    __m256 c4 = _mm256_loadu_ps(C + 4 * ldc);
    __m256 c5 = _mm256_loadu_ps(C + 5 * ldc);
    __m256 c6 = _mm256_loadu_ps(C + 6 * ldc);
    __m256 c7 = _mm256_loadu_ps(C + 7 * ldc);

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();

    // Row pointers into packed A (each row is k_rem scalars, contiguous)
    const float* a0 = A_packed + 0 * k_rem;
    const float* a1 = A_packed + 1 * k_rem;
    const float* a2 = A_packed + 2 * k_rem;
    const float* a3 = A_packed + 3 * k_rem;
    const float* a4 = A_packed + 4 * k_rem;
    const float* a5 = A_packed + 5 * k_rem;
    const float* a6 = A_packed + 6 * k_rem;
    const float* a7 = A_packed + 7 * k_rem;

    const float* b_ptr = B_packed;

    // Unrolled ×4 to reduce branch overhead and expose ILP to the scheduler.
    // Each unrolled group: 4 B loads, 32 FMAs.
    int p = 0;
    for (; p <= k_rem - 4; p += 4) {
        // Prefetch B 4 rows ahead (4×NR×4 bytes = 128 bytes = 2 cache lines)
        prefetch_l1(b_ptr + 4 * NR);

        // k = p+0
        __m256 b0 = _mm256_loadu_ps(b_ptr);  b_ptr += NR;
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[p+0]), b0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[p+0]), b0, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[p+0]), b0, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[p+0]), b0, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[p+0]), b0, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[p+0]), b0, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[p+0]), b0, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[p+0]), b0, acc7);

        // k = p+1
        __m256 b1 = _mm256_loadu_ps(b_ptr);  b_ptr += NR;
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[p+1]), b1, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[p+1]), b1, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[p+1]), b1, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[p+1]), b1, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[p+1]), b1, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[p+1]), b1, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[p+1]), b1, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[p+1]), b1, acc7);

        // k = p+2
        __m256 b2 = _mm256_loadu_ps(b_ptr);  b_ptr += NR;
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[p+2]), b2, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[p+2]), b2, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[p+2]), b2, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[p+2]), b2, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[p+2]), b2, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[p+2]), b2, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[p+2]), b2, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[p+2]), b2, acc7);

        // k = p+3
        __m256 b3 = _mm256_loadu_ps(b_ptr);  b_ptr += NR;
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[p+3]), b3, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[p+3]), b3, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[p+3]), b3, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[p+3]), b3, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[p+3]), b3, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[p+3]), b3, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[p+3]), b3, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[p+3]), b3, acc7);
    }
    // Scalar tail for remaining p iterations
    for (; p < k_rem; ++p) {
        __m256 b = _mm256_loadu_ps(b_ptr);  b_ptr += NR;
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[p]), b, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[p]), b, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[p]), b, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[p]), b, acc3);
        acc4 = _mm256_fmadd_ps(_mm256_set1_ps(a4[p]), b, acc4);
        acc5 = _mm256_fmadd_ps(_mm256_set1_ps(a5[p]), b, acc5);
        acc6 = _mm256_fmadd_ps(_mm256_set1_ps(a6[p]), b, acc6);
        acc7 = _mm256_fmadd_ps(_mm256_set1_ps(a7[p]), b, acc7);
    }

    // Apply alpha scaling (branch-free when alpha == 1.0f via FP comparison)
    if (alpha != 1.0f) {
        __m256 av = _mm256_set1_ps(alpha);
        acc0 = _mm256_mul_ps(acc0, av);  acc1 = _mm256_mul_ps(acc1, av);
        acc2 = _mm256_mul_ps(acc2, av);  acc3 = _mm256_mul_ps(acc3, av);
        acc4 = _mm256_mul_ps(acc4, av);  acc5 = _mm256_mul_ps(acc5, av);
        acc6 = _mm256_mul_ps(acc6, av);  acc7 = _mm256_mul_ps(acc7, av);
    }

    // Accumulate into C (C += alpha * acc)
    _mm256_storeu_ps(C + 0 * ldc, _mm256_add_ps(c0, acc0));
    _mm256_storeu_ps(C + 1 * ldc, _mm256_add_ps(c1, acc1));
    _mm256_storeu_ps(C + 2 * ldc, _mm256_add_ps(c2, acc2));
    _mm256_storeu_ps(C + 3 * ldc, _mm256_add_ps(c3, acc3));
    _mm256_storeu_ps(C + 4 * ldc, _mm256_add_ps(c4, acc4));
    _mm256_storeu_ps(C + 5 * ldc, _mm256_add_ps(c5, acc5));
    _mm256_storeu_ps(C + 6 * ldc, _mm256_add_ps(c6, acc6));
    _mm256_storeu_ps(C + 7 * ldc, _mm256_add_ps(c7, acc7));
}

// ─── 8×16 AVX-512 Micro-kernel ───────────────────────────────────────────────
/**
 * inner_kernel_8x16_avx512 — Compute C[0..7][0..15] += alpha * A × B
 *
 * Exact structural mirror of inner_kernel_8x8: 8 rows, ONE accumulator per
 * row, broadcast-based FMA. The only difference is register width: __m512
 * (16 floats) instead of __m256 (8 floats). Because a single ZMM register
 * already spans the full NR512=16 tile width, there is no dual-accumulator
 * split needed — unlike avx_matmul.cpp's 6×32 kernel (which needed 32-wide
 * tiles = 2 ZMM per row). This kernel has no "second half" to forget.
 *
 * Register budget (32 ZMM available): 8 accumulators + 1 B load + transient
 * broadcast = 9 in flight at any moment, 23 registers of headroom.
 */
#ifdef __AVX512F__
void inner_kernel_8x16_avx512(const float* __restrict__ A_packed,
                              const float* __restrict__ B_packed,
                              float*       __restrict__ C, int ldc,
                              int k_rem, float alpha) noexcept {
    __m512 c0 = _mm512_loadu_ps(C + 0 * ldc);
    __m512 c1 = _mm512_loadu_ps(C + 1 * ldc);
    __m512 c2 = _mm512_loadu_ps(C + 2 * ldc);
    __m512 c3 = _mm512_loadu_ps(C + 3 * ldc);
    __m512 c4 = _mm512_loadu_ps(C + 4 * ldc);
    __m512 c5 = _mm512_loadu_ps(C + 5 * ldc);
    __m512 c6 = _mm512_loadu_ps(C + 6 * ldc);
    __m512 c7 = _mm512_loadu_ps(C + 7 * ldc);

    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps();
    __m512 acc3 = _mm512_setzero_ps();
    __m512 acc4 = _mm512_setzero_ps();
    __m512 acc5 = _mm512_setzero_ps();
    __m512 acc6 = _mm512_setzero_ps();
    __m512 acc7 = _mm512_setzero_ps();

    const float* a0 = A_packed + 0 * k_rem;
    const float* a1 = A_packed + 1 * k_rem;
    const float* a2 = A_packed + 2 * k_rem;
    const float* a3 = A_packed + 3 * k_rem;
    const float* a4 = A_packed + 4 * k_rem;
    const float* a5 = A_packed + 5 * k_rem;
    const float* a6 = A_packed + 6 * k_rem;
    const float* a7 = A_packed + 7 * k_rem;

    const float* b_ptr = B_packed;

    for (int p = 0; p < k_rem; ++p) {
        __m512 b = _mm512_loadu_ps(b_ptr);  b_ptr += NR512;
        acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a0[p]), b, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a1[p]), b, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a2[p]), b, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a3[p]), b, acc3);
        acc4 = _mm512_fmadd_ps(_mm512_set1_ps(a4[p]), b, acc4);
        acc5 = _mm512_fmadd_ps(_mm512_set1_ps(a5[p]), b, acc5);
        acc6 = _mm512_fmadd_ps(_mm512_set1_ps(a6[p]), b, acc6);
        acc7 = _mm512_fmadd_ps(_mm512_set1_ps(a7[p]), b, acc7);
    }

    if (alpha != 1.0f) {
        __m512 av = _mm512_set1_ps(alpha);
        acc0 = _mm512_mul_ps(acc0, av);  acc1 = _mm512_mul_ps(acc1, av);
        acc2 = _mm512_mul_ps(acc2, av);  acc3 = _mm512_mul_ps(acc3, av);
        acc4 = _mm512_mul_ps(acc4, av);  acc5 = _mm512_mul_ps(acc5, av);
        acc6 = _mm512_mul_ps(acc6, av);  acc7 = _mm512_mul_ps(acc7, av);
    }

    _mm512_storeu_ps(C + 0 * ldc, _mm512_add_ps(c0, acc0));
    _mm512_storeu_ps(C + 1 * ldc, _mm512_add_ps(c1, acc1));
    _mm512_storeu_ps(C + 2 * ldc, _mm512_add_ps(c2, acc2));
    _mm512_storeu_ps(C + 3 * ldc, _mm512_add_ps(c3, acc3));
    _mm512_storeu_ps(C + 4 * ldc, _mm512_add_ps(c4, acc4));
    _mm512_storeu_ps(C + 5 * ldc, _mm512_add_ps(c5, acc5));
    _mm512_storeu_ps(C + 6 * ldc, _mm512_add_ps(c6, acc6));
    _mm512_storeu_ps(C + 7 * ldc, _mm512_add_ps(c7, acc7));
}
#endif  // __AVX512F__

// ─── ISA diagnostics and per-thread override ─────────────────────────────────
namespace {
enum class IsaOverride : int { Auto = 0, ForceAvx2 = 1, ForceAvx512 = 2 };
// thread_local (not a shared atomic): designed for RAII-style set/reset
// around a single call, matching the per-call `isa=` kwarg semantics —
// see the header doc comment for set_gemm_isa_override.
thread_local IsaOverride g_isa_override = IsaOverride::Auto;
}  // namespace

bool gemm_packed_avx512_hardware_available() noexcept {
#ifdef __AVX512F__
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
#else
    return false;
#endif
}

bool gemm_packed_isa_is_avx512() noexcept {
    const bool hw_avx512 = gemm_packed_avx512_hardware_available();
    switch (g_isa_override) {
        case IsaOverride::ForceAvx2:
            return false;  // always honor an explicit AVX2 request
        case IsaOverride::ForceAvx512:
            // Never claim AVX-512 if the hardware doesn't actually support
            // it — issuing AVX-512 instructions on unsupported hardware
            // would raise SIGILL. Callers wanting a hard error instead of
            // this silent, safe fallback should check
            // gemm_packed_avx512_hardware_available() themselves before
            // calling set_gemm_isa_override("avx512") — the Python binding
            // does exactly this.
            return hw_avx512;
        case IsaOverride::Auto:
        default:
            return hw_avx512;
    }
}

void set_gemm_isa_override(const char* isa) noexcept {
    if (isa == nullptr) { g_isa_override = IsaOverride::Auto; return; }
    const std::string_view sv(isa);
    if (sv == "avx2")        g_isa_override = IsaOverride::ForceAvx2;
    else if (sv == "avx512") g_isa_override = IsaOverride::ForceAvx512;
    else                     g_isa_override = IsaOverride::Auto;  // "", "auto", or unknown
}

const char* get_gemm_isa_override() noexcept {
    switch (g_isa_override) {
        case IsaOverride::ForceAvx2:   return "avx2";
        case IsaOverride::ForceAvx512: return "avx512";
        case IsaOverride::Auto:
        default:                      return "auto";
    }
}

// ─── Small-matrix direct path (no packing) ───────────────────────────────────
/**
 * sgemm_direct_avx2 — AVX2 GEMM without panel packing.
 *
 * Intended for small matrices where A, B, C together fit in L2 cache.
 * In that regime, allocating the fixed-size KC×NC packing buffer (≈2 MB
 * regardless of actual N) and filling it takes longer than the arithmetic
 * itself — sgemm_packed benchmarks at ~0.7× naive scalar at N=64.
 *
 * This function processes the output matrix row-by-row (i-loop), vectorising
 * the N dimension with AVX2 (NR=8 floats / __m256 per iteration):
 *
 *   for i in [0, M):
 *     for j in [0, N) step NR:
 *       acc[0..NR-1] = Σ_p  A[i][p] * B[p][j..j+NR-1]   (FMA, p-loop)
 *       C[i][j..j+NR-1] = alpha * acc + beta * C[i][j..j+NR-1]
 *     tail scalar for N % NR remaining columns
 *
 * B is accessed with stride ldb between k-steps — normally cache-unfriendly
 * for large matrices (hence packing). For max(M,N,K) ≤ 128, the entire B
 * fits in L2 (~64 KB at N=128, K=128), so hardware prefetch easily handles
 * the stride and there are no effective cache misses.
 *
 * beta=0.0 is handled by zeroing C in the pre-pass, matching sgemm_packed's
 * semantics (beta=0 must overwrite even NaN values in C — never multiply them).
 */
void sgemm_direct_avx2(int M, int N, int K,
                        float alpha,
                        const float* __restrict__ A, int lda,
                        const float* __restrict__ B, int ldb,
                        float beta,
                        float*       __restrict__ C, int ldc) noexcept {
    // beta pre-pass: C ← beta * C  (matches scale_matrix_c in sgemm_packed)
    if (beta != 1.0f) {
        for (int i = 0; i < M; ++i) {
            float* crow = C + static_cast<std::size_t>(i) * ldc;
            if (beta == 0.0f) {
                std::fill(crow, crow + N, 0.0f);  // explicit zero — never multiply NaN
            } else {
                for (int j = 0; j < N; ++j) crow[j] *= beta;
            }
        }
    }

    const __m256 alpha_v = _mm256_set1_ps(alpha);

    for (int i = 0; i < M; ++i) {
        const float* a_row = A + static_cast<std::size_t>(i) * lda;
        float*       c_row = C + static_cast<std::size_t>(i) * ldc;

        // ── Vectorised NR=8 columns per iteration ────────────────────────────
        // Inner k-loop: broadcast scalar a[i][p] and FMA against b[p][j..j+7].
        // B[p*ldb + j] is strided (ldb floats between k-steps); at small N
        // the full B matrix fits in L1/L2, so hardware prefetch is sufficient.
        int j = 0;
        for (; j + NR <= N; j += NR) {
            __m256 acc = _mm256_setzero_ps();
            for (int p = 0; p < K; ++p) {
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(a_row[p]),
                    _mm256_loadu_ps(B + static_cast<std::size_t>(p) * ldb + j),
                    acc);
            }
            if (alpha != 1.0f) acc = _mm256_mul_ps(acc, alpha_v);
            // C[i][j..j+7] += alpha * acc
            _mm256_storeu_ps(c_row + j,
                _mm256_add_ps(_mm256_loadu_ps(c_row + j), acc));
        }
        // ── Scalar tail for remaining N % NR columns ──────────────────────────
        for (; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p)
                sum += a_row[p] * B[static_cast<std::size_t>(p) * ldb + j];
            c_row[j] += alpha * sum;
        }
    }
}

// ─── sgemm_packed: 5-loop Goto/BLIS GEMM ─────────────────────────────────────
/**
 * sgemm_packed — Compute C = alpha*A*B + beta*C  (float32, row-major)
 *
 * Implements the standard Goto (2008) / BLIS 5-loop + 2 packing routines.
 *
 * Loop order (outermost to innermost):
 *   Loop 1 (jc): N in NC-wide tiles — parallelised with OpenMP
 *   Loop 2 (pc): K in KC-wide tiles — B packed here (L2-resident)
 *   Loop 3 (ic): M in MC-wide tiles — A packed here (L2-resident)
 *   Loop 4 (jr): nc in NR-wide tiles  } micro-kernel register tile
 *   Loop 5 (ir): mc in MR-wide tiles  }
 *
 * Packing buffers: allocated ONCE per call outside all loops, reused across
 * all panel iterations. Size is bounded by max tile dimensions (KC×NC for B,
 * MC×KC for A), regardless of actual matrix size.
 */
void sgemm_packed(int M, int N, int K,
                  float alpha,
                  const float* __restrict__ A, int lda,
                  const float* __restrict__ B, int ldb,
                  float beta,
                  float*       __restrict__ C, int ldc) {
    // K == 0 is valid (BLAS semantics): A·B is empty, so C = beta·C.
    // Both paths apply beta first and then run no K iterations.
    if (M <= 0 || N <= 0 || K < 0) return;

    // ── Small-matrix fast path: skip panel packing ────────────────────────────
    // sgemm_packed's packing buffers are fixed-size KC×NC ≈ 2 MB regardless of
    // actual N. When all dimensions fit in L2 (max(M,N,K) ≤ SMALL_GEMM_THRESHOLD),
    // allocating and filling those buffers costs more than the arithmetic:
    // measured at ~0.7× naive scalar at N=64 (see BENCHMARKS.md §Positioning vs
    // OpenBLAS). sgemm_direct_avx2 uses the original strides, achieving
    // hardware-prefetch-friendly access because the entire working set is in L2.
    //
    // Note: this bypass runs before the ISA override check. Small matrices don't
    // benefit meaningfully from AVX-512 vs AVX2 (bottleneck is memory, not FMA
    // throughput), so the direct path always uses AVX2 width regardless of the
    // isa= override value.
    if (std::max({M, N, K}) <= SMALL_GEMM_THRESHOLD) {
        sgemm_direct_avx2(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    scale_matrix_c(C, M, N, ldc, beta);

#ifdef __AVX512F__
    // ── AVX-512 8×16 dispatch path (early return) ────────────────────────────
    // Kept as a fully separate code block rather than interleaving NR/NR512
    // branches into the existing loops below — the AVX2 path below is
    // unmodified by this addition and untouched by this branch when
    // gemm_packed_isa_is_avx512() is false (non-AVX-512 hardware, or AVX-512
    // hardware detected at compile time but not at runtime — e.g. a binary
    // built with -march=native on one machine and run on another).
    if (gemm_packed_isa_is_avx512()) {
        const std::size_t b_buf_size_512 = static_cast<std::size_t>(KC) *
            static_cast<std::size_t>((NC + NR512 - 1) / NR512) * NR512;
        const std::size_t a_buf_size = static_cast<std::size_t>(MC) *
            static_cast<std::size_t>(KC);   // MR512 == MR, so A buffer is identical

#ifdef SIMD_ML_OPENMP
        struct ThreadBuffersAVX512 {
            ThreadBuffersAVX512()
                : B_packed(make_aligned_array<float>(
                      static_cast<std::size_t>(KC) *
                      static_cast<std::size_t>((NC + NR512 - 1) / NR512) * NR512)),
                  A_packed(make_aligned_array<float>(
                      static_cast<std::size_t>(MC) *
                      static_cast<std::size_t>(KC))) {}
            AlignedUniquePtr<float> B_packed;
            AlignedUniquePtr<float> A_packed;
        };
        thread_local ThreadBuffersAVX512 tls_bufs512;

#pragma omp parallel for schedule(dynamic, 1) \
    num_threads(get_num_threads()) if (get_num_threads() > 1)
        for (int jc = 0; jc < N; jc += NC) {
            int nc = std::min(N - jc, NC);
            float* B_packed = tls_bufs512.B_packed.get();
            float* A_packed = tls_bufs512.A_packed.get();

            for (int pc = 0; pc < K; pc += KC) {
                int kc = std::min(K - pc, KC);
                pack_b_panel_avx512(B + static_cast<std::size_t>(pc) * ldb + jc,
                                    ldb, kc, nc, B_packed);

                for (int ic = 0; ic < M; ic += MC) {
                    int mc = std::min(M - ic, MC);
                    pack_a_panel(A + static_cast<std::size_t>(ic) * lda + pc,
                                 lda, mc, kc, A_packed);

                    for (int jr = 0; jr < nc; jr += NR512) {
                        int nr = std::min(nc - jr, NR512);
                        const float* B_block = B_packed +
                            static_cast<std::size_t>(jr / NR512) * kc * NR512;
                        for (int ir = 0; ir < mc; ir += MR512) {
                            int mr = std::min(mc - ir, MR512);
                            const float* A_block = A_packed +
                                static_cast<std::size_t>(ir) * kc;
                            float* C_block = C + static_cast<std::size_t>(ic + ir) * ldc
                                               + jc + jr;
                            if (mr == MR512 && nr == NR512) {
                                inner_kernel_8x16_avx512(A_block, B_block, C_block, ldc, kc, alpha);
                            } else {
                                gemm_scalar_block_avx512(mr, nr, kc, A_block, B_block,
                                                         C_block, ldc, alpha);
                            }
                        }
                    }
                }
            }
        }
#else
        auto B_packed_storage = make_aligned_array<float>(b_buf_size_512);
        auto A_packed_storage = make_aligned_array<float>(a_buf_size);
        float* B_packed = B_packed_storage.get();
        float* A_packed = A_packed_storage.get();

        for (int jc = 0; jc < N; jc += NC) {
            int nc = std::min(N - jc, NC);

            for (int pc = 0; pc < K; pc += KC) {
                int kc = std::min(K - pc, KC);
                pack_b_panel_avx512(B + static_cast<std::size_t>(pc) * ldb + jc,
                                    ldb, kc, nc, B_packed);

                for (int ic = 0; ic < M; ic += MC) {
                    int mc = std::min(M - ic, MC);
                    pack_a_panel(A + static_cast<std::size_t>(ic) * lda + pc,
                                 lda, mc, kc, A_packed);

                    for (int jr = 0; jr < nc; jr += NR512) {
                        int nr = std::min(nc - jr, NR512);
                        const float* B_block = B_packed +
                            static_cast<std::size_t>(jr / NR512) * kc * NR512;
                        for (int ir = 0; ir < mc; ir += MR512) {
                            int mr = std::min(mc - ir, MR512);
                            const float* A_block = A_packed +
                                static_cast<std::size_t>(ir) * kc;
                            float* C_block = C + static_cast<std::size_t>(ic + ir) * ldc
                                               + jc + jr;
                            if (mr == MR512 && nr == NR512) {
                                inner_kernel_8x16_avx512(A_block, B_block, C_block, ldc, kc, alpha);
                            } else {
                                gemm_scalar_block_avx512(mr, nr, kc, A_block, B_block,
                                                         C_block, ldc, alpha);
                            }
                        }
                    }
                }
            }
        }
#endif  // SIMD_ML_OPENMP
        return;
    }
#endif  // __AVX512F__

    // ── AVX2 8×8 dispatch path (unchanged from before this session's AVX-512 work) ──

    // Buffer sizes (fixed, independent of actual M/N/K):
    //   B_packed: KC × ceil(NC/NR) × NR floats
    //   A_packed: MC × KC floats
    const std::size_t b_buf_size = static_cast<std::size_t>(KC) *
                                   static_cast<std::size_t>((NC + NR - 1) / NR) * NR;
    const std::size_t a_buf_size = static_cast<std::size_t>(MC) *
                                   static_cast<std::size_t>(KC);

#ifdef SIMD_ML_OPENMP
    // Thread-local buffers: each OpenMP thread gets its own A_packed/B_packed
    // so there is no false sharing on the packing writes (which are hot).
    struct ThreadBuffers {
        ThreadBuffers()
            : B_packed(make_aligned_array<float>(
                  static_cast<std::size_t>(KC) *
                  static_cast<std::size_t>((NC + NR - 1) / NR) * NR)),
              A_packed(make_aligned_array<float>(
                  static_cast<std::size_t>(MC) *
                  static_cast<std::size_t>(KC))) {}
        AlignedUniquePtr<float> B_packed;
        AlignedUniquePtr<float> A_packed;
    };
    thread_local ThreadBuffers tls_bufs;

#pragma omp parallel for schedule(dynamic, 1) \
    num_threads(get_num_threads()) if (get_num_threads() > 1)
    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(N - jc, NC);
        float* B_packed = tls_bufs.B_packed.get();
        float* A_packed = tls_bufs.A_packed.get();

        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(K - pc, KC);
            pack_b_panel(B + static_cast<std::size_t>(pc) * ldb + jc,
                         ldb, kc, nc, B_packed);

            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(M - ic, MC);
                pack_a_panel(A + static_cast<std::size_t>(ic) * lda + pc,
                             lda, mc, kc, A_packed);

                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = std::min(nc - jr, NR);
                    const float* B_block = B_packed +
                                           static_cast<std::size_t>(jr / NR) * kc * NR;
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = std::min(mc - ir, MR);
                        const float* A_block = A_packed +
                                               static_cast<std::size_t>(ir) * kc;
                        float* C_block = C + static_cast<std::size_t>(ic + ir) * ldc
                                           + jc + jr;
                        if (mr == MR && nr == NR) {
                            inner_kernel_8x8(A_block, B_block, C_block, ldc, kc, alpha);
                        } else {
                            gemm_scalar_block(mr, nr, kc, A_block, B_block,
                                              C_block, ldc, alpha);
                        }
                    }
                }
            }
        }
    }
#else
    // Single-threaded path: allocate buffers once, reuse for all panels.
    // This is the critical correctness fix: earlier code allocated inside the
    // k-loop, causing O(K/KC × N/NC) heap allocations per GEMM call.
    auto B_packed_storage = make_aligned_array<float>(b_buf_size);
    auto A_packed_storage = make_aligned_array<float>(a_buf_size);
    float* B_packed = B_packed_storage.get();
    float* A_packed = A_packed_storage.get();

    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(N - jc, NC);

        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(K - pc, KC);
            pack_b_panel(B + static_cast<std::size_t>(pc) * ldb + jc,
                         ldb, kc, nc, B_packed);

            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(M - ic, MC);
                pack_a_panel(A + static_cast<std::size_t>(ic) * lda + pc,
                             lda, mc, kc, A_packed);

                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = std::min(nc - jr, NR);
                    const float* B_block = B_packed +
                                           static_cast<std::size_t>(jr / NR) * kc * NR;
                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = std::min(mc - ir, MR);
                        const float* A_block = A_packed +
                                               static_cast<std::size_t>(ir) * kc;
                        float* C_block = C + static_cast<std::size_t>(ic + ir) * ldc
                                           + jc + jr;
                        if (mr == MR && nr == NR) {
                            inner_kernel_8x8(A_block, B_block, C_block, ldc, kc, alpha);
                        } else {
                            gemm_scalar_block(mr, nr, kc, A_block, B_block,
                                              C_block, ldc, alpha);
                        }
                    }
                }
            }
        }
    }
#endif  // SIMD_ML_OPENMP
}

// ─── Thread count control ─────────────────────────────────────────────────────
static std::atomic_int g_num_threads{1};

void set_num_threads(int n) noexcept {
    if (n <= 0) n = 1;
    g_num_threads.store(n, std::memory_order_relaxed);
#ifdef SIMD_ML_OPENMP
    omp_set_num_threads(n);
#endif
}

int get_num_threads() noexcept {
#ifdef SIMD_ML_OPENMP
    int t = g_num_threads.load(std::memory_order_relaxed);
    return (t <= 0) ? omp_get_max_threads() : t;
#else
    return 1;
#endif
}

}  // namespace gemm
}  // namespace simd_ml
