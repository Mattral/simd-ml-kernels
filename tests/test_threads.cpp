/**
 * test_threads.cpp — thread-count control and the threaded packed GEMM.
 *
 *   1. set_num_threads(n) / get_num_threads() round-trip; n <= 0 clamps to 1.
 *   2. gemm_chunks cuts a column block into chunks that tile it exactly, in
 *      whole micro-panels, no taller than MC, no narrower than 128 columns
 *      unless the block is.
 *   3. gemm_threads: 1 for small products and inside a parallel region.
 *   4. Every table: the result with 2, 3, 4 and 8 threads is bitwise equal to
 *      the 1-thread result (each C element is computed by one thread in the
 *      same order), on shapes that cross MC, KC, NC, the micro-tile edges, the
 *      transposes, alpha/beta and strides.
 *   5. GEMMs called concurrently from the threads of an enclosing OpenMP
 *      region (each runs single-threaded, on its own buffers) match the
 *      sequential results bitwise.
 * The GEMM suite itself (test_gemm_packed.cpp) runs with 1, 2 and 4 threads.
 */

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include "../src/kernels/gemm/avx2_gemm_packed.hpp"
#include "kernel_tables.hpp"

#ifdef SIMD_ML_OPENMP
#  include <omp.h>
#endif

namespace {

using simd_ml::gemm::get_num_threads;
using simd_ml::gemm::set_num_threads;

bool report(bool pass, const char* what) {
    std::printf("  %-66s %s\n", what, pass ? "PASS" : "FAIL");
    return pass;
}

void fill(std::vector<float>& v, unsigned seed) {
    unsigned s = seed;
    for (float& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (float)(int(s >> 16) % 2000 - 1000) * 1e-3f;
    }
}

bool test_set_get() {
    bool ok = true;
    set_num_threads(1);
    ok &= report(get_num_threads() == 1, "set_num_threads(1) -> get_num_threads() == 1");
    set_num_threads(0);
    ok &= report(get_num_threads() == 1, "set_num_threads(0) clamps to 1");
#ifdef SIMD_ML_OPENMP
    set_num_threads(3);
    ok &= report(get_num_threads() == 3, "set_num_threads(3) -> 3 (OpenMP)");
#else
    set_num_threads(3);
    ok &= report(get_num_threads() == 1, "set_num_threads(3) -> 1 (built without OpenMP)");
#endif
    return ok;
}

bool test_chunks() {
    bool ok = true;
    int bad = 0, cases = 0;
    for (int threads : {1, 2, 3, 4, 8, 16})
        for (int M : {1, 5, 6, 7, 64, 97, 431, 1000, 2048})
            for (int N : {1, 17, 100, 128, 129, 1000, 2720})
                for (const auto& t : {std::make_pair(6, 16), std::make_pair(12, 32)}) {
                    const int mr = t.first, nr = t.second, mc_max = 426;
                    const simd_ml::gemm::GemmChunks c =
                        simd_ml::gemm::gemm_chunks(threads, M, N, mr, nr, mc_max);
                    ++cases;
                    const int n_full = (N + nr - 1) / nr * nr;
                    bool good = c.mc > 0 && c.nc > 0 && c.mc % mr == 0 && c.nc % nr == 0 &&
                                c.mc <= mc_max &&
                                (long long)c.m_chunks * c.mc >= M && (long long)(c.m_chunks - 1) * c.mc < M &&
                                (long long)c.n_chunks * c.nc >= N && (long long)(c.n_chunks - 1) * c.nc < N &&
                                c.nc >= std::min(n_full, std::max(nr, 128 / nr * nr));
                    // At least 2 MR-row micro-panels per thread: no column split.
                    if ((M + mr - 1) / mr >= 2 * threads && c.n_chunks != 1) good = false;
                    if (!good && bad++ < 5)
                        std::printf("    threads=%d M=%d N=%d tile=%dx%d: mc=%d nc=%d chunks=%dx%d\n",
                                    threads, M, N, mr, nr, c.mc, c.nc, c.m_chunks, c.n_chunks);
                }
    char msg[96];
    std::snprintf(msg, sizeof msg, "gemm_chunks tiles every block exactly (%d cases)", cases);
    ok &= report(bad == 0, msg);
    return ok;
}

bool test_thread_count() {
    bool ok = true;
    set_num_threads(4);
    ok &= report(simd_ml::gemm::gemm_threads(8, 8, 8, 6, 16) == 1, "gemm_threads(8x8x8) == 1");
#ifdef SIMD_ML_OPENMP
    ok &= report(simd_ml::gemm::gemm_threads(512, 512, 512, 6, 16) == 4,
                 "gemm_threads(512^3) == 4 with set_num_threads(4)");
    int inner = -1;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        inner = simd_ml::gemm::gemm_threads(512, 512, 512, 6, 16);
    }
    ok &= report(omp_get_max_threads() < 2 || inner == 1, "gemm_threads == 1 inside a parallel region");
#else
    ok &= report(simd_ml::gemm::gemm_threads(512, 512, 512, 6, 16) == 1,
                 "gemm_threads(512^3) == 1 without OpenMP");
#endif
    return ok;
}

struct Shape { int M, N, K; bool ta = false, tb = false; float alpha = 1.0f, beta = 0.0f; int pad = 0; };

/** C after one call of the table's sgemm with `threads` threads (C starts as fill(seed 3)). */
std::vector<float> run(const simd_ml::KernelTable& t, const Shape& s, int threads) {
    const int lda = (s.ta ? s.M : s.K) + s.pad, ldb = (s.tb ? s.K : s.N) + s.pad, ldc = s.N + s.pad;
    std::vector<float> A((std::size_t)(s.ta ? s.K : s.M) * lda), B((std::size_t)(s.tb ? s.N : s.K) * ldb),
        C((std::size_t)s.M * ldc);
    fill(A, 1); fill(B, 2); fill(C, 3);
    set_num_threads(threads);
    t.sgemm(s.ta, s.tb, s.M, s.N, s.K, s.alpha, A.data(), lda, B.data(), ldb, s.beta, C.data(), ldc);
    return C;
}

bool test_bitwise(const simd_ml::KernelTable& t) {
    int kc = 0, mc = 0, nc = 0;
    t.sgemm_blocking(&kc, &mc, &nc);
    if (kc == 0) {  // the scalar table has no packed path and never threads
        std::printf("  bitwise across thread counts: SKIP (%s table is single-threaded)\n", t.name);
        return true;
    }
    std::vector<Shape> shapes = {
        {512, 512, 512}, {mc + 7, 300, 2 * kc + 3}, {13, 3000, 200}, {7, 4096, 300},
        {1000, 37, 300}, {181, 175, 177}, {300, nc + 17, 23}, {97, 2 * nc + 1, 41},
        {190, 210, 230, true, false}, {190, 210, 230, false, true}, {190, 210, 230, true, true},
        {64, 1024, 1024, false, true}, {200, 190, 180, false, false, 2.5f, 0.3f},
        {190, 210, 230, false, false, 1.0f, 0.0f, 9}, {23, 3000, 129, true, true, -0.5f, 1.0f, 3},
    };
    bool ok = true;
    for (const Shape& s : shapes) {
        const std::vector<float> ref = run(t, s, 1);
        int differ = 0;
        for (int threads : {2, 3, 4, 8}) {
            const std::vector<float> got = run(t, s, threads);
            if (std::memcmp(got.data(), ref.data(), ref.size() * sizeof(float)) != 0) differ = threads;
        }
        char msg[96];
        std::snprintf(msg, sizeof msg, "%5dx%5dx%5d %c%c a=%.1f b=%.1f pad=%d: 2/3/4/8 threads == 1 thread%s",
                      s.M, s.N, s.K, s.ta ? 'T' : 'N', s.tb ? 'T' : 'N', s.alpha, s.beta, s.pad,
                      differ ? " (differs)" : "");
        ok &= report(differ == 0, msg);
    }
    return ok;
}

bool test_concurrent_callers(const simd_ml::KernelTable& t) {
#ifdef SIMD_ML_OPENMP
    const Shape shapes[4] = {{200, 210, 230}, {431, 64, 300}, {97, 2000, 41, false, true}, {300, 300, 300, true}};
    std::vector<float> ref[4], got[4];
    for (int i = 0; i < 4; ++i) ref[i] = run(t, shapes[i], 1);
    set_num_threads(4);   // each caller is inside a parallel region: runs single-threaded
#pragma omp parallel for num_threads(4) schedule(static, 1)
    for (int i = 0; i < 4; ++i) {
        const Shape& s = shapes[i];
        const int lda = s.ta ? s.M : s.K, ldb = s.tb ? s.K : s.N;
        std::vector<float> A((std::size_t)(s.ta ? s.K : s.M) * lda), B((std::size_t)(s.tb ? s.N : s.K) * ldb),
            C((std::size_t)s.M * s.N);
        fill(A, 1); fill(B, 2); fill(C, 3);
        t.sgemm(s.ta, s.tb, s.M, s.N, s.K, s.alpha, A.data(), lda, B.data(), ldb, s.beta, C.data(), s.N);
        got[i] = std::move(C);
    }
    bool same = true;
    for (int i = 0; i < 4; ++i) same &= got[i] == ref[i];
    return report(same, "4 concurrent GEMMs from an omp parallel for == sequential");
#else
    (void)t;
    std::printf("  concurrent callers: SKIP (built without OpenMP)\n");
    return true;
#endif
}

}  // namespace

int run_thread_tests() {
    std::printf("\n── Thread count and threaded GEMM ──\n");
    const int original = get_num_threads();
    bool ok = test_set_get();
    ok &= test_chunks();
    ok &= test_thread_count();
    ok &= for_each_kernel_table([](const simd_ml::KernelTable& t) {
        const bool bitwise = test_bitwise(t);
        const bool concurrent = test_concurrent_callers(t);
        return bitwise && concurrent;
    });
    set_num_threads(original);
    std::printf("  Overall: %s\n", ok ? "ALL PASS" : "SOME FAILURES");
    return ok ? 0 : 1;
}
