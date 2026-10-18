#pragma once

/**
 * cpuid.hpp — CPU feature detection from CPUID and XGETBV.
 *
 * Used once, by dispatch.cpp, to choose the kernel table (dispatch/kernels.hpp).
 *
 * ─── CPUID bits used ──────────────────────────────────────────────────────────
 * Leaf 1, ECX:      12 FMA, 20 SSE4.2, 27 OSXSAVE (XGETBV usable), 28 AVX, 29 F16C
 * Leaf 7.0, EBX:     5 AVX2, 16 AVX-512F, 17 AVX-512DQ, 30 AVX-512BW, 31 AVX-512VL
 * Leaf 7.0, ECX:    11 AVX512_VNNI
 * Leaf 7.1, EAX:     4 AVX-VNNI (VEX-encoded vpdpbusd on YMM), 5 AVX512_BF16
 *
 * XGETBV(0) returns XCR0: the OS must save a register file's state on context
 * switches, or using it faults even when CPUID reports the instructions. YMM
 * needs XCR0 bits 1-2 (0x06); ZMM additionally bits 5-7 (opmask, ZMM0-15 upper
 * halves, ZMM16-31): 0xE6. Every has_* flag below that needs YMM or ZMM state
 * already includes the OS check, so a caller only tests the one flag.
 *
 * Non-x86 targets: every flag is false and the scalar kernels are used.
 */

#include <cstdint>
#include <vector>

namespace simd_ml {
namespace dispatch {

struct CpuFeatures {
    bool has_sse42      = false;
    bool has_fma        = false;  ///< FMA3 and OS YMM state
    bool has_avx2       = false;  ///< AVX2 and OS YMM state
    bool has_f16c       = false;  ///< F16C and OS YMM state
    bool has_avx512f    = false;  ///< AVX-512F and OS ZMM state
    bool has_avx512dq   = false;  ///< AVX-512DQ and OS ZMM state
    bool has_avx512bw   = false;  ///< AVX-512BW and OS ZMM state
    bool has_avx512vl   = false;  ///< AVX-512VL and OS ZMM state
    bool has_avx512bf16 = false;  ///< AVX512_BF16 + AVX-512F and OS ZMM state
    bool has_avx_vnni   = false;  ///< AVX-VNNI + AVX2 and OS YMM state (Alder Lake, Lunar Lake, Sapphire Rapids)
    bool has_avx512vnni = false;  ///< AVX512_VNNI + AVX-512F and OS ZMM state (Cascade Lake and later)

    /**
     * detect() — run CPUID/XGETBV. Implemented in dispatch.cpp; cheap, but
     * callers should use cpu_features() (dispatch/kernels.hpp), which caches.
     * All false on non-x86 targets.
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

/** GEMM block sizes for an MR×NR micro-kernel (see gemm_blocking). */
struct GemmBlocking {
    int kc, mc, nc;
    unsigned l1d_kib, l2_kib, llc_kib;  ///< the caches they were derived from (0: unknown)
};

/**
 * gemm_blocking — KC, MC, NC for an MR×NR FP32 micro-kernel from the caches of
 * the calling core (detect_caches; Goto/BLIS rules, Low et al. 2016):
 *   KC = L1D / 2 / (NR·4)    a KC×NR B micro-panel fills half of L1D
 *   MC = L2 / 4 / (KC·4)     the packed MC×KC A block fills a quarter of L2
 *   NC = LLC / 2 / (KC·4)    the packed KC×NC B panel fills half of the LLC
 * rounded down to multiples of 8, MR and NR and clamped to [64, 1024],
 * [MR, 2048], [NR, 8192]. Unknown caches: 32 KiB L1D, 256 KiB L2, 8 MiB LLC.
 * Measurements behind the fractions: avx2_gemm_packed.cpp header.
 */
GemmBlocking gemm_blocking(int mr, int nr) noexcept;

}  // namespace dispatch
}  // namespace simd_ml
