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
 *   6. Decode GEMV: gemv_slice tiles [0, N) in whole 8-row groups, balanced
 *      to one group; gemv_threads respects the bytes-per-thread minimum, the
 *      row groups and enclosing parallel regions; every table's FP32/FP16/
 *      BF16 x·Wᵀ (M = 1 and 3) and gemv_q8_0 / gemv_q4_0 give bitwise the
 *      1-thread result with 2, 3, 4 and 8 threads (minimum lowered so small
 *      shapes split; N with every remainder mod 8).
 * The GEMM suite itself (test_gemm_packed.cpp) runs with 1, 2 and 4 threads.
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include "../src/kernels/gemm/avx2_gemm_packed.hpp"
#include "../src/kernels/quant/q8_0.hpp"
#include "../src/kernels/quant/q4_0.hpp"
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

bool test_gemv_slices() {
    using simd_ml::gemm::gemv_slice;
    using simd_ml::gemm::kGemvRowGroup;
    int bad = 0, cases = 0;
    for (int team = 1; team <= 16; ++team)
        for (int N : {1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 100, 512, 1003, 2048, 8193, 128256}) {
            ++cases;
            int prev = 0, smallest = N, largest = 0;
            bool good = true;
            for (int t = 0; t < team; ++t) {
                const simd_ml::gemm::GemvSlice s = gemv_slice(N, t, team);
                good &= s.lo == prev && s.hi >= s.lo && s.lo % kGemvRowGroup == 0 &&
                        (s.hi % kGemvRowGroup == 0 || s.hi == N);
                smallest = std::min(smallest, s.hi - s.lo);
                largest = std::max(largest, s.hi - s.lo);
                prev = s.hi;
            }
            good &= prev == N;
            const int groups = (N + kGemvRowGroup - 1) / kGemvRowGroup;
            if (groups >= team) good &= largest - smallest <= kGemvRowGroup;   // within one group
            if (!good && bad++ < 5) std::printf("    team=%d N=%d\n", team, N);
        }
    char msg[96];
    std::snprintf(msg, sizeof msg, "gemv_slice tiles [0, N) in whole 8-row groups (%d cases)", cases);
    return report(bad == 0, msg);
}

bool test_gemv_thread_count() {
    using simd_ml::gemm::gemv_threads;
    bool ok = true;
    set_num_threads(4);
    ok &= report(gemv_threads(4096, 1000.0) == 1, "gemv_threads(1 kB of weights) == 1");
#ifdef SIMD_ML_OPENMP
    ok &= report(gemv_threads(4096, 4096.0 * 4096 * 18 / 32) == 4,
                 "gemv_threads(Q4_0 4096x4096) == 4 with set_num_threads(4)");
    ok &= report(gemv_threads(17, 1e9) == 3, "gemv_threads(N = 17) == 3 (row groups)");
    int inner = -1;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        inner = gemv_threads(4096, 1e9);
    }
    ok &= report(omp_get_max_threads() < 2 || inner == 1, "gemv_threads == 1 inside a parallel region");
#else
    ok &= report(gemv_threads(4096, 1e9) == 1, "gemv_threads == 1 without OpenMP");
#endif
    return ok;
}

/** y from one decode GEMV of every kind (FP32 M=1 and M=3, FP16, BF16, Q8_0, Q4_0) with `threads`. */
std::vector<float> run_gemv(const simd_ml::KernelTable& t, int N, int K, int threads) {
    std::vector<float> x(3 * (std::size_t)K), w((std::size_t)N * K);
    fill(x, 5); fill(w, 6);
    std::vector<std::uint16_t> x16(x.size()), w16(w.size()), xb(x.size()), wb(w.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        x16[i] = simd_ml::quant::fp32_to_fp16(x[i]);
        std::uint32_t u; std::memcpy(&u, &x[i], 4); xb[i] = (std::uint16_t)(u >> 16);
    }
    for (std::size_t i = 0; i < w.size(); ++i) {
        w16[i] = simd_ml::quant::fp32_to_fp16(w[i]);
        std::uint32_t u; std::memcpy(&u, &w[i], 4); wb[i] = (std::uint16_t)(u >> 16);
    }
    std::vector<std::uint8_t> q8(simd_ml::quant::q8_0_row_bytes(K) * N), q4((std::size_t)N * (K / 32) * 18);
    simd_ml::quant::quantize_q8_0_rows(w.data(), N, K, q8.data());
    simd_ml::quant::quantize_q4_0_rows(w.data(), N, K, q4.data());
    set_num_threads(threads);
    std::vector<float> out, y(3 * (std::size_t)N);
    const auto take = [&](std::size_t n) { out.insert(out.end(), y.begin(), y.begin() + n); };
    // Outputs that must be overwritten start from values that differ per
    // thread count, so a row no thread computed cannot compare equal.
    const auto stale = [&]() { fill(y, 100u + (unsigned)threads); };
    fill(y, 7);   // beta = 0.5 reads C: the same start for every count
    t.sgemm(false, true, 1, N, K, 1.0f, x.data(), K, w.data(), K, 0.5f, y.data(), N); take(N);
    stale();
    t.sgemm(false, true, 3, N, K, -1.5f, x.data(), K, w.data(), K, 0.0f, y.data(), N); take(3 * (std::size_t)N);
    stale();
    t.sgemm_f16(true, 1, N, K, 1.0f, x16.data(), K, w16.data(), K, 0.0f, y.data(), N); take(N);
    stale();
    t.sgemm_bf16(true, 2, N, K, 1.0f, xb.data(), K, wb.data(), K, 0.0f, y.data(), N); take(2 * (std::size_t)N);
    stale();
    t.gemv_q8_0(N, K, x.data(), q8.data(), y.data()); take(N);
    stale();
    t.gemv_q4_0(N, K, x.data(), q4.data(), y.data()); take(N);
    return out;
}

bool test_gemv_bitwise(const simd_ml::KernelTable& t) {
    // Lower the minimum so that every shape below splits across the team.
    const double saved = simd_ml::gemm::set_gemv_min_bytes_per_thread(1.0);
    bool ok = true;
    for (const auto& nk : {std::make_pair(1, 64), std::make_pair(9, 32), std::make_pair(31, 96),
                           std::make_pair(63, 4096), std::make_pair(100, 160), std::make_pair(257, 1024),
                           std::make_pair(1003, 288), std::make_pair(2048, 64), std::make_pair(4101, 128)}) {
        const int N = nk.first, K = nk.second;
        const std::vector<float> ref = run_gemv(t, N, K, 1);
        int differ = 0;
        for (int threads : {2, 3, 4, 8})
            if (run_gemv(t, N, K, threads) != ref) differ = threads;
        char msg[112];
        std::snprintf(msg, sizeof msg, "GEMV %5dx%5d fp32/f16/bf16/q8_0/q4_0: 2/3/4/8 threads == 1 thread%s",
                      N, K, differ ? " (differs)" : "");
        ok &= report(differ == 0, msg);
    }
    simd_ml::gemm::set_gemv_min_bytes_per_thread(saved);
    return ok;
}

}  // namespace

int run_thread_tests() {
    std::printf("\n── Thread count and threaded GEMM ──\n");
    const int original = get_num_threads();
    bool ok = test_set_get();
    ok &= test_chunks();
    ok &= test_thread_count();
    ok &= test_gemv_slices();
    ok &= test_gemv_thread_count();
    ok &= for_each_kernel_table([](const simd_ml::KernelTable& t) {
        const bool bitwise = test_bitwise(t);
        const bool concurrent = test_concurrent_callers(t);
        const bool gemv = test_gemv_bitwise(t);
        return bitwise && concurrent && gemv;
    });
    set_num_threads(original);
    std::printf("  Overall: %s\n", ok ? "ALL PASS" : "SOME FAILURES");
    return ok ? 0 : 1;
}
