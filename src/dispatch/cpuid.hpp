#pragma once

/**
 * cpuid.hpp — Lightweight CPUID-based CPU feature detection.
 *
 * Used by gemm_dispatcher.cpp to select the best available kernel
 * implementation at process startup (via std::call_once).
 *
 * ─── CPUID leaves used ────────────────────────────────────────────────────────
 * Leaf 1, ECX:
 *   bit 12 — FMA3          (Haswell+)    → required for _mm256_fmadd_ps
 *   bit 20 — SSE4.2        (Nehalem+)    → scalar fallback gate
 *   bit 27 — OSXSAVE       → XGETBV is usable (needed to check OS register state)
 *   bit 29 — F16C          (Ivy Bridge+) → sgemm_f16 (VEX-encoded, needs OS AVX state)
 *
 * Leaf 7 / sub-leaf 0, EBX:
 *   bit  5 — AVX2          (Haswell+)    → primary SIMD path
 *   bit 16 — AVX-512F      (Skylake-SP+) → detected but not yet used (see ROADMAP.md §v0.8)
 *   bit 17 — AVX-512DQ     (Skylake-SP+) → same
 *
 * Leaf 7 / sub-leaf 1, EAX:
 *   bit  5 — AVX512_BF16   (Cooper Lake, Sapphire Rapids, Zen 4) → vdpbf16ps
 *
 * XGETBV(0) / XCR0: the OS must save the register state an ISA uses, or its
 * instructions fault even when CPUID reports them. AVX needs XCR0 bits 1-2
 * (0x06); AVX-512 additionally needs bits 5-7 (opmask, ZMM0-15 upper, ZMM16-31),
 * 0xE6 in total. has_f16c and has_avx512bf16 include these OS checks.
 *
 * ─── Thread safety ────────────────────────────────────────────────────────────
 * CpuFeatures::detect() is called exactly once via std::call_once in
 * gemm_dispatcher.cpp.  The struct itself contains only scalar bools and
 * is safe to read concurrently after initialisation.
 *
 * ─── Non-x86 platforms ───────────────────────────────────────────────────────
 * On non-x86 targets (ARM, RISC-V, etc.) all feature flags remain false
 * and the dispatch layer falls back to naive_sgemm.  Adding ARM NEON/SVE
 * detection is tracked in ROADMAP.md.
 */

#include <cstdint>
#include <vector>

namespace simd_ml {
namespace dispatch {

struct CpuFeatures {
    bool has_sse42    = false;
    bool has_fma      = false;
    bool has_avx2     = false;
    bool has_avx512f  = false;
    bool has_avx512dq = false;
    bool has_f16c       = false;  ///< F16C and OS AVX state enabled
    bool has_avx512bf16 = false;  ///< AVX512_BF16 + AVX-512F and OS ZMM state enabled

    /**
     * detect() — probe CPUID and return a populated CpuFeatures struct.
     * Implemented in gemm_dispatcher.cpp (requires cpuid inline asm).
     * Returns all-false on non-x86 platforms.
     */
    static CpuFeatures detect() noexcept;
};

/**
 * CacheLevel — one data/instruction/unified cache as reported by the CPU for
 * the core the calling thread runs on (CPUID leaf 4 on Intel, 0x8000001D on
 * AMD). On hybrid CPUs (e.g. Lunar Lake) P- and E-cores report different
 * caches, so pin the thread first if the answer matters.
 */
struct CacheLevel {
    int      level;             ///< 1, 2, 3 ...
    char     type;              ///< 'D' data, 'I' instruction, 'U' unified
    unsigned size_kib;
    unsigned ways;
    unsigned line_bytes;
    unsigned max_sharing_cpus;  ///< upper bound on logical CPUs sharing it (CPUID field)
};

/** detect_caches — empty on non-x86 or CPUs without these leaves. */
std::vector<CacheLevel> detect_caches();

}  // namespace dispatch
}  // namespace simd_ml
