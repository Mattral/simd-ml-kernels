/**
 * test_gemm_packed.cpp — FP32 GEMM (KernelTable::sgemm) of every table the CPU
 * supports, against a float64 reference.
 *
 * Bound, elementwise (as in selftest): |C − C64| ≤ γ·(|α|·(|A||B|)ij + |β|·|C0ij|),
 * γ = 2·(K+2)·u, u = 2^-24, where C64 is α·AB + β·C0 in double.
 *
 * Shapes are chosen from the table's own tile (6×16 on AVX2) and block sizes
 * (sgemm_blocking: KC/MC/NC derived from this CPU's caches), so every edge is
 * crossed by one: M, N around tile multiples; K, M, N just past KC, MC, NC.
 * Strided cases pad A, B and C with NaN: a read of A/B padding or a write to
 * C padding shows up as NaN in C or as a changed sentinel.
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include "../src/kernels/gemm/avx2_gemm_packed.hpp"
#include "../src/kernels/gemm/naive_gemm.hpp"
#include "../src/kernels/cache_alloc.hpp"
#include "kernel_tables.hpp"

// The table the test functions below call into (set by the runner).
static const simd_ml::KernelTable* T = nullptr;

using GemmFn = void (*)(int, int, int, float, const float*, int, const float*, int, float,
                        float*, int);

static void fill_random(float* buf, std::size_t n, unsigned seed) {
    unsigned s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (float)(int(s >> 16) % 2000 - 1000) * 1e-3f;
    }
}

struct Case {
    int M, N, K;
    float alpha = 1.0f, beta = 0.0f;
    int pad_a = 0, pad_b = 0, pad_c = 0;  // extra floats per row (lda = K + pad_a, ...)
    int offset = 0;                       // floats added to every base pointer (unaligned)
};

// Runs fn on the case; returns the worst |err| / bound (pass: <= 1) or 1e30 if
// a padding sentinel of C changed.
static double run_case(GemmFn fn, const Case& c) {
    const int lda = c.K + c.pad_a, ldb = c.N + c.pad_b, ldc = c.N + c.pad_c;
    const float nan = std::nanf("");
    std::vector<float> A(c.offset + (std::size_t)c.M * lda + 1, nan),
        B(c.offset + (std::size_t)c.K * ldb + 1, nan), C(c.offset + (std::size_t)c.M * ldc + 16, nan);
    float* a = A.data() + c.offset; float* b = B.data() + c.offset; float* cc = C.data() + c.offset;
    std::vector<float> tmp((std::size_t)std::max(1, c.M * std::max(c.K, c.N)));
    fill_random(tmp.data(), (std::size_t)c.M * c.K, 1);
    for (int i = 0; i < c.M; ++i) std::copy_n(&tmp[(std::size_t)i * c.K], c.K, a + (std::size_t)i * lda);
    std::vector<float> tb((std::size_t)c.K * c.N);
    fill_random(tb.data(), tb.size(), 2);
    for (int p = 0; p < c.K; ++p) std::copy_n(&tb[(std::size_t)p * c.N], c.N, b + (std::size_t)p * ldb);
    std::vector<float> c0((std::size_t)c.M * c.N);
    fill_random(c0.data(), c0.size(), 3);
    if (c.beta == 0.0f) std::fill(c0.begin(), c0.end(), nan);   // beta = 0 must not read C
    for (int i = 0; i < c.M; ++i) std::copy_n(&c0[(std::size_t)i * c.N], c.N, cc + (std::size_t)i * ldc);

    fn(c.M, c.N, c.K, c.alpha, a, lda, b, ldb, c.beta, cc, ldc);

    const double gamma = 2.0 * (c.K + 2) * std::ldexp(1.0, -24);
    double worst = 0.0;
    for (int i = 0; i < c.M; ++i)
        for (int j = 0; j < c.N; ++j) {
            double ref = 0.0, mag = 0.0;
            for (int p = 0; p < c.K; ++p) {
                const double x = (double)a[(std::size_t)i * lda + p] * b[(std::size_t)p * ldb + j];
                ref += x; mag += std::fabs(x);
            }
            const double cin = c.beta == 0.0f ? 0.0 : c0[(std::size_t)i * c.N + j];
            ref = c.alpha * ref + c.beta * cin;
            const double bound = gamma * (std::fabs(c.alpha) * mag + std::fabs(c.beta * cin)) + 1e-30;
            const double got = cc[(std::size_t)i * ldc + j];
            const double r = std::isnan(got) ? 1e30 : std::fabs(got - ref) / bound;
            worst = std::max(worst, r);
        }
    // Padding of C (columns N..ldc of every row, and past the last row) is untouched.
    for (int i = 0; i < c.M; ++i)
        for (int j = c.N; j < ldc; ++j)
            if (!std::isnan(cc[(std::size_t)i * ldc + j])) return 1e30;
    for (int j = 0; j < 16; ++j)
        if (!std::isnan(cc[(std::size_t)c.M * ldc + j])) return 1e30;
    return worst;
}

static bool check(GemmFn fn, const char* what, const Case& c) {
    const double w = run_case(fn, c);
    const bool pass = w <= 1.0;
    printf("  %-7s %5dx%5dx%5d a=%.1f b=%.1f pad=%d/%d/%d off=%d  err/bound=%-8.3g %s\n", what,
           c.M, c.N, c.K, c.alpha, c.beta, c.pad_a, c.pad_b, c.pad_c, c.offset, w,
           pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_table_gemm() {
    int kc = 0, mc = 0, nc = 0;
    T->sgemm_blocking(&kc, &mc, &nc);
    printf(" %s: tile %s, blocking KC=%d MC=%d NC=%d\n", T->name, T->sgemm_tile, kc, mc, nc);
    bool ok = true;
    const GemmFn fn = T->sgemm;

    printf(" Square sizes:\n");
    for (int s : {1, 2, 5, 6, 7, 8, 15, 16, 17, 31, 64, 127, 128, 129, 161, 255, 256, 512})
        ok &= check(fn, "square", {s, s, s});

    printf(" Every row/column tail of the 6x16 tile (direct and packed sizes):\n");
    for (int M : {1, 5, 6, 7, 11, 12, 13})
        for (int N : {1, 8, 15, 16, 17, 33})
            ok &= check(fn, "tails", {M, N, 37});
    for (int M : {175, 181})
        for (int N : {161, 175})
            ok &= check(fn, "tails", {M, N, 177});

    if (kc > 0) {
        printf(" Block boundaries (KC=%d, MC=%d, NC=%d):\n", kc, mc, nc);
        for (int K : {kc - 1, kc, kc + 1, 2 * kc + 3})
            ok &= check(fn, "kc", {97, 113, K});
        for (int M : {mc - 1, mc, mc + 1, mc + 7})
            ok &= check(fn, "mc", {M, 33, 41});
        ok &= check(fn, "nc", {13, nc + 17, 23});
        ok &= check(fn, "nc", {29, 2 * nc + 1, 19});
    }

    printf(" Skinny and wide:\n");
    ok &= check(fn, "skinny", {1, 4096, 300});
    ok &= check(fn, "skinny", {300, 1, 300});
    ok &= check(fn, "skinny", {300, 300, 1});

    printf(" alpha/beta (beta = 0 with C = NaN must overwrite):\n");
    for (float alpha : {1.0f, 2.5f, -0.5f})
        for (float beta : {0.0f, 1.0f, 0.3f}) {
            ok &= check(fn, "ab", {33, 41, 27, alpha, beta});
            ok &= check(fn, "ab", {200, 190, 180, alpha, beta});
        }

    printf(" Strides (NaN padding) and unaligned pointers:\n");
    for (int off : {0, 1}) {
        ok &= check(fn, "stride", {23, 37, 29, 1.0f, 0.5f, 3, 5, 7, off});
        ok &= check(fn, "stride", {190, 210, 230, 1.0f, 0.0f, 1, 16, 9, off});
    }
    return ok;
}

// The direct (unpacked) AVX2 path on its own, including shapes the dispatcher
// sends to the packed path, so both paths are checked on the same shapes.
static bool test_direct_avx2() {
    if (!simd_ml::kernels_for(simd_ml::Isa::avx2)) {
        printf(" sgemm_direct_avx2: SKIP (CPU has no AVX2)\n");
        return true;
    }
    printf(" sgemm_direct_avx2 (called directly):\n");
    const GemmFn fn = &simd_ml::gemm::sgemm_direct_avx2;
    bool ok = true;
    for (const Case& c : {Case{7, 17, 300}, Case{257, 129, 65}, Case{64, 64, 64, 2.0f, 0.5f},
                          Case{37, 53, 41, 1.0f, 0.0f, 2, 3, 5, 1}})
        ok &= check(fn, "direct", c);
    return ok;
}

// K == 0 (BLAS semantics): A·B is an empty sum, so C = beta·C; beta == 0
// must overwrite C even if it holds NaN. Direct-path and packed-path shapes,
// plus the scalar reference naive_sgemm.
static bool test_k_zero_applies_beta() {
    bool all_pass = true;
    for (int M : {7, 300}) {
        const int N = (M == 7) ? 9 : 400;
        for (float beta : {0.0f, 0.5f}) {
            for (int impl = 0; impl < 2; ++impl) {
                auto C = make_aligned_array<float>(static_cast<std::size_t>(M) * N);
                const float init = (beta == 0.0f) ? std::nanf("") : 3.0f;
                std::fill(C.get(), C.get() + M * N, init);
                const float dummy = 1.0f;  // A and B are never read when K == 0
                if (impl == 0)
                    T->sgemm(M, N, 0, 1.0f, &dummy, 1, &dummy, N, beta, C.get(), N);
                else
                    naive_sgemm(M, N, 0, 1.0f, &dummy, 1, &dummy, N, beta, C.get(), N);
                const float expect = (beta == 0.0f) ? 0.0f : beta * 3.0f;
                bool pass = true;
                for (int i = 0; i < M * N; ++i) pass &= (C[i] == expect);
                printf("  K=0 %-12s M=%-3d beta=%.1f  C==%.1f: %s\n",
                       impl == 0 ? T->name : "naive_sgemm", M, beta, expect,
                       pass ? "PASS" : "FAIL");
                all_pass &= pass;
            }
        }
    }
    return all_pass;
}

int run_gemm_packed_tests() {
    printf("\n── FP32 GEMM Correctness Tests ──\n");
    bool all_pass = for_each_kernel_table([](const simd_ml::KernelTable& t) {
        T = &t;
        bool ok = test_table_gemm();
        printf(" Edge: K=0 gives C = beta*C:\n");
        ok &= test_k_zero_applies_beta();
        return ok;
    });
    all_pass &= test_direct_avx2();
    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
