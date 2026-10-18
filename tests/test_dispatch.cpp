/**
 * test_dispatch.cpp — the kernel-table selection rule (dispatch/kernels.hpp).
 *
 * Checks the decisions, not the kernels (the other test files run the kernels
 * of every table): that kernels() is the widest table kernels_for() offers
 * (or the one SIMD_ML_ISA caps it at), that the CPU features each table
 * needs imply the narrower tables, and that every table is complete.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../src/dispatch/kernels.hpp"

using simd_ml::Isa;

static bool check(bool cond, const char* what) {
    std::printf("  %-64s %s\n", what, cond ? "PASS" : "FAIL");
    return cond;
}

int run_dispatch_tests() {
    std::printf("\n── Kernel table dispatch ──\n");
    const auto& f = simd_ml::cpu_features();
    std::printf("  CPU: avx2=%d fma=%d f16c=%d avx512f=%d dq=%d bw=%d vl=%d bf16=%d\n",
                f.has_avx2, f.has_fma, f.has_f16c, f.has_avx512f, f.has_avx512dq,
                f.has_avx512bw, f.has_avx512vl, f.has_avx512bf16);
    bool ok = true;

    const simd_ml::KernelTable* widest = nullptr;
    for (int i = 0; i < simd_ml::kNumIsa; ++i) {
        const Isa isa = static_cast<Isa>(i);
        const simd_ml::KernelTable* t = simd_ml::kernels_for(isa);
        std::printf("  %-7s compiled=%d usable=%d\n", simd_ml::isa_name(isa),
                    simd_ml::isa_compiled(isa), t != nullptr);
        if (!t) continue;
        widest = t;
        char what[96];
        std::snprintf(what, sizeof what, "%s table: isa and name match, every entry set", t->name);
        ok &= check(t->isa == isa && std::strcmp(t->name, simd_ml::isa_name(isa)) == 0 &&
                    t->sgemm && t->sgemm_f16 && t->sgemm_bf16 && t->sgemm_u8s8 && t->gelu_erf && t->gelu_tanh &&
                    t->relu && t->silu && t->softmax_row && t->layer_norm_row, what);
    }
    ok &= check(simd_ml::kernels_for(Isa::scalar) != nullptr, "scalar table is always usable");
    ok &= check(!simd_ml::kernels_for(Isa::avx512) || simd_ml::kernels_for(Isa::avx2),
                "avx512 usable implies avx2 usable");
    ok &= check(!simd_ml::kernels_for(Isa::avx2) || (f.has_avx2 && f.has_fma && f.has_f16c),
                "avx2 usable implies AVX2, FMA, F16C and OS YMM state");
    ok &= check(!simd_ml::kernels_for(Isa::avx512) ||
                (f.has_avx512f && f.has_avx512dq && f.has_avx512bw && f.has_avx512vl),
                "avx512 usable implies AVX-512F/DQ/BW/VL and OS ZMM state");

    // kernels() is the widest usable table unless SIMD_ML_ISA caps it.
    const char* env = std::getenv("SIMD_ML_ISA");
    const simd_ml::KernelTable& active = simd_ml::kernels();
    std::printf("  active table: %s (SIMD_ML_ISA=%s)\n", active.name, env ? env : "unset");
    if (!env) {
        ok &= check(&active == widest, "kernels() is the widest usable table");
    } else {
        ok &= check(static_cast<int>(active.isa) <= static_cast<int>(widest->isa),
                    "kernels() is not wider than the widest usable table");
    }
    std::printf("  Overall: %s\n", ok ? "ALL PASS" : "SOME FAILURES");
    return ok ? 0 : 1;
}
