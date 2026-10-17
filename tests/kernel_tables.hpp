#pragma once

/**
 * kernel_tables.hpp — run a test on every kernel table this CPU supports.
 *
 * The test binary is built for baseline x86-64 and reaches the AVX2 and
 * AVX-512 kernels only through the tables, so each test runs on the scalar
 * table everywhere, on AVX2 where the CPU has it, and on AVX-512 under Intel
 * SDE (-skx/-spr) or on an AVX-512 machine.
 */

#include <cstdio>
#include <vector>

#include "../src/dispatch/kernels.hpp"

inline std::vector<const simd_ml::KernelTable*> kernel_tables_to_test() {
    std::vector<const simd_ml::KernelTable*> out;
    for (int i = 0; i < simd_ml::kNumIsa; ++i)
        if (const simd_ml::KernelTable* t = simd_ml::kernels_for(static_cast<simd_ml::Isa>(i)))
            out.push_back(t);
    return out;
}

/** body(table) for each supported table; true if every call returned true. */
template <typename Body>
bool for_each_kernel_table(Body body) {
    bool ok = true;
    for (const simd_ml::KernelTable* t : kernel_tables_to_test()) {
        std::printf(" [%s kernels]\n", t->name);
        ok &= body(*t);
    }
    return ok;
}
