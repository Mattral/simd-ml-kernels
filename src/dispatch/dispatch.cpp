/**
 * dispatch.cpp — CPU feature detection and the choice of kernel table.
 *
 * Compiled with the baseline flags only (no -mavx*), like the bindings: code
 * here runs before anything has checked what the CPU supports. See
 * kernels.hpp for the ISA units and the selection rule.
 *
 * Also defines the C++ entry points that dispatch through the table
 * (gemm::sgemm_packed and friends) and the thread-count control, so that
 * nothing outside an ISA unit has to be compiled with that ISA's flags.
 */

#include "cpuid.hpp"
#include "kernels.hpp"
#include "../kernels/gemm/avx2_gemm_packed.hpp"
#include "../kernels/gemm/f16_gemm.hpp"
#include "../kernels/gemm/bf16_gemm.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef SIMD_ML_OPENMP
#  include <omp.h>
#endif

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#  define SIMD_ML_X86_ASM 1
#endif

namespace simd_ml {
namespace dispatch {

#ifdef SIMD_ML_X86_ASM
static void cpuid_ex(unsigned leaf, unsigned subleaf,
                     unsigned& a, unsigned& b, unsigned& c, unsigned& d) noexcept {
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(leaf), "c"(subleaf));
}

// XGETBV with ECX = 0 returns XCR0. Only valid when CPUID.1:ECX.OSXSAVE is set.
static unsigned long long read_xcr0() noexcept {
    unsigned lo = 0, hi = 0;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#endif

CpuFeatures CpuFeatures::detect() noexcept {
    CpuFeatures f;
#ifdef SIMD_ML_X86_ASM
    unsigned a, b, c, d;
    cpuid_ex(0, 0, a, b, c, d);
    const unsigned max_leaf = a;
    cpuid_ex(1, 0, a, b, c, d);
    const bool osxsave = (c & (1u << 27)) != 0;
    const bool avx     = (c & (1u << 28)) != 0;
    const unsigned long long xcr0 = osxsave ? read_xcr0() : 0;
    const bool os_ymm = avx && (xcr0 & 0x06) == 0x06;
    const bool os_zmm = os_ymm && (xcr0 & 0xE6) == 0xE6;
    f.has_sse42 = (c & (1u << 20)) != 0;
    f.has_fma   = os_ymm && (c & (1u << 12)) != 0;
    f.has_f16c  = os_ymm && (c & (1u << 29)) != 0;
    if (max_leaf >= 7) {
        cpuid_ex(7, 0, a, b, c, d);
        const unsigned max_subleaf = a;
        f.has_avx2     = os_ymm && (b & (1u << 5)) != 0;
        f.has_avx512f  = os_zmm && (b & (1u << 16)) != 0;
        f.has_avx512dq = os_zmm && (b & (1u << 17)) != 0;
        f.has_avx512bw = os_zmm && (b & (1u << 30)) != 0;
        f.has_avx512vl = os_zmm && (b & (1u << 31)) != 0;
        if (max_subleaf >= 1) {
            cpuid_ex(7, 1, a, b, c, d);
            f.has_avx512bf16 = f.has_avx512f && (a & (1u << 5)) != 0;
        }
    }
#endif
    return f;
}

std::vector<CacheLevel> detect_caches() {
    std::vector<CacheLevel> out;
#ifdef SIMD_ML_X86_ASM
    unsigned a, b, c, d;
    cpuid_ex(0, 0, a, b, c, d);
    const unsigned max_leaf = a;
    const bool intel = b == 0x756e6547u && d == 0x49656e69u && c == 0x6c65746eu;  // "GenuineIntel"
    const bool amd   = b == 0x68747541u && d == 0x69746e65u && c == 0x444d4163u;  // "AuthenticAMD"
    unsigned leaf = 0;
    if (intel && max_leaf >= 4) {
        leaf = 4;
    } else if (amd) {
        cpuid_ex(0x80000000u, 0, a, b, c, d);
        if (a >= 0x8000001Du) leaf = 0x8000001Du;
    }
    if (leaf == 0) return out;
    // Same register layout in both leaves (Intel SDM vol. 2A, CPUID leaf 04H;
    // AMD APM vol. 3, CPUID Fn8000_001D).
    for (unsigned i = 0; i < 16; ++i) {
        cpuid_ex(leaf, i, a, b, c, d);
        const unsigned type = a & 0x1f;   // 0 = no more caches
        if (type == 0) break;
        CacheLevel cl;
        cl.level            = static_cast<int>((a >> 5) & 7);
        cl.type             = type == 1 ? 'D' : type == 2 ? 'I' : 'U';
        cl.ways             = (b >> 22) + 1;
        const unsigned partitions = ((b >> 12) & 0x3ff) + 1;
        cl.line_bytes       = (b & 0xfff) + 1;
        const unsigned sets = c + 1;
        cl.size_kib         = cl.ways * partitions * cl.line_bytes * sets / 1024;
        cl.max_sharing_cpus = ((a >> 14) & 0xfff) + 1;
        out.push_back(cl);
    }
#endif
    return out;
}

GemmBlocking gemm_blocking(int mr, int nr) noexcept {
    unsigned l1d = 0, l2 = 0, llc = 0;
    try {
        for (const CacheLevel& c : detect_caches()) {
            if (c.type == 'I') continue;
            if (c.level == 1) l1d = c.size_kib;
            else if (c.level == 2) l2 = c.size_kib;
            if (c.level >= 2) llc = std::max(llc, c.size_kib);  // no L3: the L2 is the LLC
        }
    } catch (...) {  // std::bad_alloc from the vector: keep the defaults
    }
    GemmBlocking b{};
    b.l1d_kib = l1d; b.l2_kib = l2; b.llc_kib = llc;
    const long long L1 = (l1d ? l1d : 32) * 1024LL, L2 = (l2 ? l2 : 256) * 1024LL,
                    LLC = (llc ? llc : 8192) * 1024LL;
    auto round_down = [](long long v, int m) { return v / m * m; };
    const long long kc = std::clamp(round_down(L1 / 2 / (nr * 4LL), 8), 64LL, 1024LL);
    const long long mc = std::clamp(round_down(L2 / 4 / (kc * 4), mr), (long long)mr, 2048LL / mr * mr);
    const long long nc = std::clamp(round_down(LLC / 2 / (kc * 4), nr), (long long)nr, 8192LL / nr * nr);
    b.kc = static_cast<int>(kc);
    b.mc = static_cast<int>(mc);
    b.nc = static_cast<int>(nc);
    return b;
}

}  // namespace dispatch

// ─── Table selection ──────────────────────────────────────────────────────────

const dispatch::CpuFeatures& cpu_features() noexcept {
    static const dispatch::CpuFeatures f = dispatch::CpuFeatures::detect();
    return f;
}

bool isa_compiled(Isa isa) noexcept {
    switch (isa) {
        case Isa::scalar: return true;
#ifdef SIMD_ML_HAVE_AVX2
        case Isa::avx2:   return true;
#endif
#ifdef SIMD_ML_HAVE_AVX512
        case Isa::avx512: return true;
#endif
        default:          return false;
    }
}

const char* isa_name(Isa isa) noexcept {
    switch (isa) {
        case Isa::scalar: return "scalar";
        case Isa::avx2:   return "avx2";
        case Isa::avx512: return "avx512";
    }
    return "unknown";
}

// What each table's units were compiled for. The AVX2 units use F16C too
// (sgemm_f16), and the compiler may emit BW/VL instructions in the AVX-512
// units, so those are required even where the source does not use them.
static bool cpu_runs(Isa isa) noexcept {
    const dispatch::CpuFeatures& f = cpu_features();
    const bool avx2 = f.has_avx2 && f.has_fma && f.has_f16c;
    switch (isa) {
        case Isa::scalar: return true;
        case Isa::avx2:   return avx2;
        case Isa::avx512: return avx2 && f.has_avx512f && f.has_avx512dq &&
                                 f.has_avx512bw && f.has_avx512vl;
    }
    return false;
}

const KernelTable* kernels_for(Isa isa) noexcept {
    if (!isa_compiled(isa) || !cpu_runs(isa)) return nullptr;
    switch (isa) {
        case Isa::scalar: return &kernel_table_scalar();
#ifdef SIMD_ML_HAVE_AVX2
        case Isa::avx2:   return &kernel_table_avx2();
#endif
#ifdef SIMD_ML_HAVE_AVX512
        case Isa::avx512: return &kernel_table_avx512();
#endif
        default:          return nullptr;
    }
}

static const KernelTable& select_table() noexcept {
    int cap = kNumIsa - 1;
    if (const char* env = std::getenv("SIMD_ML_ISA")) {
        for (int i = 0; i < kNumIsa; ++i)
            if (std::strcmp(env, isa_name(static_cast<Isa>(i))) == 0) cap = i;
    }
    for (int i = cap; i > 0; --i)
        if (const KernelTable* t = kernels_for(static_cast<Isa>(i))) return *t;
    return kernel_table_scalar();
}

// The active table. Set once from select_table() on first use; use_kernels()
// may replace it (tests). One atomic load per kernel call.
static std::atomic<const KernelTable*> g_active{nullptr};

const KernelTable& kernels() noexcept {
    const KernelTable* t = g_active.load(std::memory_order_acquire);
    if (t == nullptr) {
        static const KernelTable& selected = select_table();   // thread-safe init
        const KernelTable* expected = nullptr;
        g_active.compare_exchange_strong(expected, &selected, std::memory_order_acq_rel);
        t = g_active.load(std::memory_order_acquire);
    }
    return *t;
}

bool use_kernels(Isa isa) noexcept {
    const KernelTable* t = kernels_for(isa);
    if (t == nullptr) return false;
    g_active.store(t, std::memory_order_release);
    return true;
}

// ─── Dispatching C++ entry points ─────────────────────────────────────────────

namespace gemm {

void sgemm_packed(int M, int N, int K, float alpha,
                  const float* A, int lda, const float* B, int ldb,
                  float beta, float* C, int ldc) {
    kernels().sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

bool f16c_available() noexcept { return cpu_features().has_f16c; }

bool bf16_avx512bf16_available() noexcept { return cpu_features().has_avx512bf16; }

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
    const int t = g_num_threads.load(std::memory_order_relaxed);
    return t <= 0 ? omp_get_max_threads() : t;
#else
    return 1;
#endif
}

}  // namespace gemm
}  // namespace simd_ml
