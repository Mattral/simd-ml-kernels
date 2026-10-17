/**
 * test_gemm_packed.cpp — Correctness tests for avx2_gemm_packed (8×8 micro-kernel).
 *
 * Tests KernelTable::sgemm of every table the CPU supports (scalar, AVX2 8x8,
 * AVX-512 8x16) against the scalar triple-loop reference (naive_sgemm) across:
 *   - Square sizes (1, 7, 8, 9, 16, 127, 128, 129, 256, 512, 1024)
 *   - Non-square / non-power-of-two sizes (edge/tail handling)
 *   - Alpha/beta scaling paths
 *   - beta=0 overwrite correctness (C must not accumulate prior garbage)
 *
 * Tolerance: 1e-4 absolute. For larger K the error can reach ~K×ε_mach;
 * the 1e-4 cap is generous enough to cover K=1024 (worst case ~1024×1.2e-7 ≈ 1.2e-4).
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "../src/kernels/gemm/avx2_gemm_packed.hpp"
#include "../src/kernels/gemm/naive_gemm.hpp"
#include "../src/kernels/cache_alloc.hpp"
#include "kernel_tables.hpp"

// The table the test functions below call into (set by the runner).
static const simd_ml::KernelTable* T = nullptr;

static void fill_random(float* buf, int n, unsigned seed = 42) {
    unsigned s = seed;
    for (int i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (float)(int(s >> 16) % 200 - 100) * 0.01f;
    }
}

static bool test_sgemm_packed(int M, int N, int K,
                               float alpha = 1.0f, float beta = 0.0f,
                               unsigned seed_c = 0) {
    auto A     = make_aligned_array<float>(static_cast<std::size_t>(M) * K);
    auto B     = make_aligned_array<float>(static_cast<std::size_t>(K) * N);
    auto C_ref = make_aligned_array<float>(static_cast<std::size_t>(M) * N);
    auto C_got = make_aligned_array<float>(static_cast<std::size_t>(M) * N);

    fill_random(A.get(), M * K, 1);
    fill_random(B.get(), K * N, 2);
    if (seed_c != 0) {
        fill_random(C_ref.get(), M * N, seed_c);
        std::copy(C_ref.get(), C_ref.get() + M * N, C_got.get());
    } else {
        std::fill(C_ref.get(), C_ref.get() + M * N, 0.0f);
        std::fill(C_got.get(), C_got.get() + M * N, 0.0f);
    }

    // Reference: naive GEMM (alpha/beta handled inline)
    {
        auto AB = make_aligned_array<float>(static_cast<std::size_t>(M) * N);
        naive_sgemm(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, AB.get(), N);
        for (int i = 0; i < M * N; ++i)
            C_ref[i] = alpha * AB[i] + beta * C_ref[i];
    }

    T->sgemm(M, N, K, alpha, A.get(), K, B.get(), N, beta, C_got.get(), N);

    float max_err = 0.0f;
    int   worst   = -1;
    for (int i = 0; i < M * N; ++i) {
        float e = std::fabs(C_got[i] - C_ref[i]);
        if (e > max_err) { max_err = e; worst = i; }
    }

    bool pass = (max_err < 1e-4f);
    if (alpha == 1.0f && beta == 0.0f) {
        printf("  [%4d×%4d×%4d]  max_abs=%.2e  %s\n",
               M, N, K, max_err, pass ? "PASS" : "FAIL");
    } else {
        printf("  [%4d×%4d×%4d] α=%.1f β=%.1f  max_abs=%.2e  %s\n",
               M, N, K, alpha, beta, max_err, pass ? "PASS" : "FAIL");
    }
    if (!pass && worst >= 0) {
        printf("    worst at [%d,%d]: got=%.6f  ref=%.6f\n",
               worst / N, worst % N, C_got[worst], C_ref[worst]);
    }
    return pass;
}

static bool test_beta_zero_overwrites_poison() {
    // beta=0 must completely overwrite C, not accumulate into garbage
    const int N = 32;
    auto A = make_aligned_array<float>(N * N);
    auto B = make_aligned_array<float>(N * N);
    auto C = make_aligned_array<float>(N * N);
    fill_random(A.get(), N * N, 20);
    fill_random(B.get(), N * N, 21);
    std::fill(C.get(), C.get() + N * N, 1e30f);  // poison

    T->sgemm(N, N, N, 1.0f, A.get(), N, B.get(), N, 0.0f, C.get(), N);

    auto C_ref = make_aligned_array<float>(N * N);
    naive_sgemm(N, N, N, 1.0f, A.get(), N, B.get(), N, 0.0f, C_ref.get(), N);

    float max_err = 0.0f;
    for (int i = 0; i < N * N; ++i)
        max_err = std::max(max_err, std::fabs(C[i] - C_ref[i]));

    bool pass = (max_err < 1e-4f);
    printf("  beta=0 overwrites poison    max_abs=%.2e  %s\n", max_err, pass ? "PASS" : "FAIL");
    return pass;
}

// K == 0 (BLAS semantics): A·B is an empty sum, so C = beta·C; beta == 0
// must overwrite C even if it holds NaN. Covers the direct path (M, N <= 128)
// and the packed path (M > 128), plus the scalar reference naive_sgemm.
static bool test_k_zero_applies_beta() {
    bool all_pass = true;
    for (int M : {7, 300}) {
        const int N = 9;
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

/**
 * test_gemm_packed_avx512_full_coverage — Forces 100% of the output through
 * the full-block micro-kernel path, with ZERO edge/tail blocks possible,
 * regardless of which ISA sgemm_packed's runtime dispatch selects.
 *
 * MR512 == MR (both 8), so M only needs to be a multiple of 8 to guarantee
 * full row coverage on either path. N must be a common multiple of NR=8
 * (AVX2) and NR512=16 (AVX-512) so that whichever kernel actually runs,
 * every output column comes from a full-width micro-kernel call — none can
 * be silently "saved" by the scalar edge-block fallback masking a bug in
 * the wide kernel.
 *
 * Unlike avx_matmul.cpp's 6×32 kernel, inner_kernel_8x16_avx512 has no
 * dual-accumulator split (one ZMM spans the full NR512=16 width), so there
 * is no "second half never written" failure mode possible by construction.
 * The residual risk this test targets is dispatch wiring: wrong buffer
 * sizing, wrong packed-block indexing, or routing to the wrong kernel/stride
 * combination — any of which would produce either wrong values or an
 * out-of-bounds access (also why this path is additionally verified under
 * AddressSanitizer — see .github/workflows/ci.yml sanitizers job).
 */
static bool test_gemm_packed_avx512_full_coverage() {
    const int M = 80;    // = 10 * MR(8) = 10 * MR512(8): zero remainder rows, either ISA
    const int N = 160;   // = 20 * NR(8) = 10 * NR512(16): zero remainder cols, either ISA
    const int K = 97;    // arbitrary, not a multiple of anything convenient

    auto A     = make_aligned_array<float>(static_cast<std::size_t>(M) * K);
    auto B     = make_aligned_array<float>(static_cast<std::size_t>(K) * N);
    auto C_ref = make_aligned_array<float>(static_cast<std::size_t>(M) * N);
    auto C_got = make_aligned_array<float>(static_cast<std::size_t>(M) * N);

    fill_random(A.get(), M * K, /*seed=*/33);
    fill_random(B.get(), K * N, /*seed=*/44);
    std::fill(C_ref.get(), C_ref.get() + M * N, 0.0f);
    std::fill(C_got.get(), C_got.get() + M * N, 0.0f);

    naive_sgemm(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C_ref.get(), N);
    T->sgemm(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C_got.get(), N);

    float max_err = 0.0f;
    for (int i = 0; i < M * N; ++i)
        max_err = std::max(max_err, std::fabs(C_got[i] - C_ref[i]));

    bool pass = (max_err < 1e-4f);
    printf("  AVX-512 full-coverage [%dx%dx%d]  isa=%s  max_abs=%.2e  %s\n",
           M, N, K,
           T->sgemm_tile,
           max_err, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_small_gemm_direct_path() {
    // ── Small sizes (<= SMALL_GEMM_THRESHOLD): the AVX2/AVX-512 tables run the
    // unpacked sgemm_direct_avx2 path here; the scalar table its only path.
    bool all_pass = true;
    struct Case { int M, N, K; float alpha, beta; const char* label; };
    const Case cases[] = {
        {  1,   8,   1, 1.0f, 0.0f, "1×8×1 (tiny)"},
        {  8,   8,   8, 1.0f, 0.0f, "8×8×8 (one full tile)"},
        {  7,   9,  11, 1.0f, 0.0f, "7×9×11 (all tails)"},
        { 16,  16,  16, 1.0f, 0.0f, "16×16×16"},
        { 64,  64,  64, 1.0f, 0.0f, "64×64×64 (key regression)"},
        {128, 128, 128, 1.0f, 0.0f, "128×128×128 (at threshold)"},
        { 64,  32,  48, 2.5f, 0.5f, "64×32×48 α=2.5 β=0.5"},
        { 33,  41,  27, 0.5f, 1.0f, "33×41×27 α=0.5 β=1.0"},
    };
    for (const auto& c : cases) {
        auto A     = make_aligned_array<float>(static_cast<std::size_t>(c.M) * c.K);
        auto B     = make_aligned_array<float>(static_cast<std::size_t>(c.K) * c.N);
        auto C_ref = make_aligned_array<float>(static_cast<std::size_t>(c.M) * c.N);
        auto C_got = make_aligned_array<float>(static_cast<std::size_t>(c.M) * c.N);

        fill_random(A.get(),     c.M * c.K, 11);
        fill_random(B.get(),     c.K * c.N, 22);
        fill_random(C_ref.get(), c.M * c.N, 33);
        std::copy(C_ref.get(), C_ref.get() + c.M * c.N, C_got.get());

        naive_sgemm(c.M, c.N, c.K,
                    c.alpha, A.get(), c.K, B.get(), c.N,
                    c.beta,  C_ref.get(), c.N);

        T->sgemm(c.M, c.N, c.K,
                 c.alpha, A.get(), c.K, B.get(), c.N,
                 c.beta,  C_got.get(), c.N);

        float max_err = 0.0f;
        for (int i = 0; i < c.M * c.N; ++i)
            max_err = std::max(max_err, std::fabs(C_got[i] - C_ref[i]));

        bool pass = (max_err < 1e-4f);
        printf("  direct %-30s  max_abs=%.2e  %s\n",
               c.label, max_err, pass ? "PASS" : "FAIL");
        all_pass &= pass;
    }

    // ── Dispatch check: sgemm_packed on small inputs must match naive ────────
    // If the dispatch is broken (direct path not reached), the packed path
    // can still produce numerically correct results for these sizes — so this
    // is a dispatch-neutral correctness check, not a strict dispatch test.
    printf("  dispatch (sgemm_packed on small N via existing test suite):\n");
    for (int s : {8, 16, 32, 64, 128})
        all_pass &= test_sgemm_packed(s, s, s);

    return all_pass;
}

int run_gemm_packed_tests() {
    printf("\n── Packed GEMM Correctness Tests ──\n");
    const bool all_pass = for_each_kernel_table([](const simd_ml::KernelTable& t) {
        T = &t;
        bool ok = true;

        printf(" Standard sizes (α=1, β=0):\n");
        for (int s : {1, 7, 8, 9, 16, 127, 128, 129, 256, 512, 1024})
            ok &= test_sgemm_packed(s, s, s);

        printf(" Non-square edge cases:\n");
        ok &= test_sgemm_packed(1,   4096, 4096);
        ok &= test_sgemm_packed(13,  31,   37);
        ok &= test_sgemm_packed(32,  7,    17);

        printf(" Alpha/beta scaling:\n");
        ok &= test_sgemm_packed(32, 32, 32, 2.5f, 0.3f, /*seed_c=*/99);
        ok &= test_sgemm_packed(64, 64, 64, 0.5f, 1.0f, /*seed_c=*/77);

        printf(" Edge: beta=0 must overwrite poisoned C:\n");
        ok &= test_beta_zero_overwrites_poison();

        printf(" Edge: K=0 gives C = beta*C:\n");
        ok &= test_k_zero_applies_beta();

        printf(" AVX-512 dual-kernel dispatch regression guard:\n");
        ok &= test_gemm_packed_avx512_full_coverage();

        printf(" Small-matrix direct path (SMALL_GEMM_THRESHOLD=%d, v0.9):\n",
               simd_ml::gemm::SMALL_GEMM_THRESHOLD);
        ok &= test_small_gemm_direct_path();
        return ok;
    });

    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
