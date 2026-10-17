/**
 * gemm_dispatcher.cpp — Runtime Kernel Dispatcher
 *
 * Detects CPU features at startup (once, via std::call_once) and populates
 * the KernelRegistry with the best available implementations.
 *
 * Decision tree:
 *   AVX2 + FMA  → simd_ml::gemm::sgemm_packed + activations::gelu_avx2.
 *                 sgemm_packed has its OWN internal runtime dispatch between
 *                 an 8×16 AVX-512 micro-kernel and an 8×8 AVX2 micro-kernel
 *                 (see avx2_gemm_packed.cpp: gemm_packed_isa_is_avx512()).
 *                 The isa_label below reflects that internal choice, queried
 *                 once here (CPU features don't change at runtime, so a
 *                 single query at registry-population time is equivalent to
 *                 querying on every call, without the per-call overhead).
 *   SSE4.2      → naive scalar GEMM (SSE GEMM not yet implemented)
 *   fallback    → naive scalar GEMM
 *
 * This file also implements CpuFeatures::detect() via inline CPUID.
 */

#include "../../dispatch/cpuid.hpp"
#include "../../dispatch/kernel_registry.hpp"
#include "../../kernels/activations/activations.hpp"
#include "naive_gemm.hpp"
#include "avx2_gemm_packed.hpp"

#include <mutex>

namespace {
    static std::once_flag g_registry_once;
    static simd_ml::dispatch::KernelRegistry g_registry;
}

// CpuFeatures::detect() implementation
namespace simd_ml { namespace dispatch {

static void cpuid_ex(int leaf, int subleaf,
                     unsigned int& a, unsigned int& b,
                     unsigned int& c, unsigned int& d) {
    a = b = c = d = 0;
#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("cpuid"
                     : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
                     : "a"(leaf), "c"(subleaf));
#endif
#endif
}

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
// XGETBV with ECX=0 returns XCR0. Only valid when CPUID.1:ECX.OSXSAVE is set.
static unsigned long long read_xcr0() noexcept {
    unsigned int lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#endif

CpuFeatures CpuFeatures::detect() noexcept {
    CpuFeatures f;
#if defined(__x86_64__) || defined(__i386__)
    unsigned int a, b, c, d;
    cpuid_ex(0, 0, a, b, c, d);
    const unsigned int max_leaf = a;
    // Leaf 1: FMA (ECX bit 12), SSE4.2 (bit 20), OSXSAVE (bit 27), F16C (bit 29)
    cpuid_ex(1, 0, a, b, c, d);
    f.has_fma   = (c & (1u << 12)) != 0;
    f.has_sse42 = (c & (1u << 20)) != 0;
    const bool osxsave = (c & (1u << 27)) != 0;
    const bool f16c    = (c & (1u << 29)) != 0;
#if defined(__GNUC__) || defined(__clang__)
    const unsigned long long xcr0 = osxsave ? read_xcr0() : 0;
#else
    const unsigned long long xcr0 = 0;
#endif
    const bool os_avx    = (xcr0 & 0x06) == 0x06;
    const bool os_avx512 = (xcr0 & 0xE6) == 0xE6;
    f.has_f16c = f16c && os_avx;
    // Leaf 7, sub-leaf 0: AVX2 (EBX bit 5), AVX-512F (EBX bit 16), AVX-512DQ (EBX bit 17)
    if (max_leaf >= 7) {
        cpuid_ex(7, 0, a, b, c, d);
        const unsigned int max_subleaf = a;
        f.has_avx2    = (b & (1u << 5))  != 0;
        f.has_avx512f = (b & (1u << 16)) != 0;
        f.has_avx512dq= (b & (1u << 17)) != 0;
        // Leaf 7, sub-leaf 1: AVX512_BF16 (EAX bit 5)
        if (max_subleaf >= 1) {
            cpuid_ex(7, 1, a, b, c, d);
            f.has_avx512bf16 = (a & (1u << 5)) != 0 && f.has_avx512f && os_avx512;
        }
    }
#endif
    return f;
}

} } // namespace simd_ml::dispatch

namespace simd_ml { namespace dispatch {

const KernelRegistry& get_kernels() noexcept {
    std::call_once(g_registry_once, [] {
        CpuFeatures f = CpuFeatures::detect();

        // Note: sgemm_packed (avx2_gemm_packed.cpp) has its own internal
        // AVX-512/AVX2 dispatch. Query it directly so isa_label reflects
        // reality rather than a separately-maintained (and easily stale)
        // duplicate of that decision.
        if (f.has_avx2 && f.has_fma) {
            g_registry.sgemm     = &gemm::sgemm_packed;
            g_registry.gelu      = &activations::gelu_avx2;  // global namespace
            g_registry.isa_label = gemm::gemm_packed_isa_is_avx512() ? "avx512" : "avx2";
        } else if (f.has_sse42) {
            g_registry.sgemm     = &gemm_ref::naive_sgemm;
            g_registry.gelu      = nullptr;
            g_registry.isa_label = "sse42";
        } else {
            g_registry.sgemm     = &gemm_ref::naive_sgemm;
            g_registry.gelu      = nullptr;
            g_registry.isa_label = "scalar";
        }
    });
    return g_registry;
}

void sgemm(int M, int N, int K,
           float alpha, const float* A, int lda,
           const float* B, int ldb,
           float beta, float* C, int ldc) {
    const KernelRegistry& reg = get_kernels();
    if (reg.sgemm) {
        reg.sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

const char* detected_isa() noexcept {
    return get_kernels().isa_label;
}

} } // namespace simd_ml::dispatch
