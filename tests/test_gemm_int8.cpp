/**
 * test_gemm_int8.cpp — INT8 GEMM (KernelTable::sgemm_u8s8, gemm/int8_gemm.hpp)
 * of every table the CPU supports, and each of its paths called directly
 * (FP32 conversion, AVX-VNNI, AVX512_VNNI) where the CPU and build have it,
 * against a float64 reference:
 *
 *   C64[i][j] = a_scale · b_scales[j] · Σ_k (A[i][k] − zp) · op(B)[k][j]   (exact)
 *   |C − C64| ≤ 2·(K+2)·u·|a_scale·b_scales[j]|·Σ_k |A − zp|·|op(B)|,  u = 2^-24
 *
 * Shapes cross the INT8 block sizes (KC = 4 × the FP32 KC, in k-quads: K at
 * multiples of 4 ± 1..3, KC ± 1), MC, NC and the 6×16 / 12×32 tiles; trans_b
 * (B stored N×K); padded leading dimensions with sentinel padding in C; base
 * pointers offset by one byte; zero points 0, 128 and 255. The extreme case
 * (A = 255, B = −128, zp = 0, K = KC) checks that the int32 block sums do not
 * overflow.
 */

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

#include "../src/kernels/gemm/avx2_gemm_packed.hpp"
#include "../src/kernels/gemm/int8_gemm.hpp"
#include "kernel_tables.hpp"

using U8S8Fn = simd_ml::GemmU8S8Fn;

struct Case {
    int M, N, K;
    bool tb = false;
    int zp = 128;
    int pad_a = 0, pad_b = 0, pad_c = 0;  // extra elements per row
    int offset = 0;                       // bytes / floats added to every base pointer
    int fill = 0;                         // 0: random; 1: A = 255, B = -128 (int32 overflow check)
};

static unsigned next(unsigned& s) { s = s * 1664525u + 1013904223u; return s >> 8; }

// Returns the worst |err| / bound (pass: <= 1), or 1e30 if C's padding changed.
static double run_case(U8S8Fn fn, const Case& c) {
    const int b_rows = c.tb ? c.N : c.K, b_cols = c.tb ? c.K : c.N;
    const int lda = c.K + c.pad_a, ldb = b_cols + c.pad_b, ldc = c.N + c.pad_c;
    std::vector<std::uint8_t> A(c.offset + (std::size_t)c.M * lda + 1, 0xA5);
    std::vector<std::int8_t> B(c.offset + (std::size_t)b_rows * ldb + 1, 0x5A);
    const float sentinel = -7.25f;
    std::vector<float> C(c.offset + (std::size_t)c.M * ldc + 16, sentinel);
    std::vector<float> scales(c.N + 1);
    std::uint8_t* a = A.data() + c.offset;
    std::int8_t* b = B.data() + c.offset;
    float* cc = C.data() + c.offset;
    unsigned s = 12345u + static_cast<unsigned>(c.M * 31 + c.N * 7 + c.K);
    for (int i = 0; i < c.M; ++i)
        for (int p = 0; p < c.K; ++p) a[(std::size_t)i * lda + p] = c.fill ? 255 : static_cast<std::uint8_t>(next(s));
    for (int r = 0; r < b_rows; ++r)
        for (int q = 0; q < b_cols; ++q) b[(std::size_t)r * ldb + q] = c.fill ? -128 : static_cast<std::int8_t>(next(s));
    for (int j = 0; j < c.N; ++j) scales[j] = 1e-3f * (1 + next(s) % 1000) * ((next(s) & 1) ? 1.0f : -1.0f);
    const float a_scale = 0.0123f;
    auto b_at = [&](int p, int j) { return c.tb ? b[(std::size_t)j * ldb + p] : b[(std::size_t)p * ldb + j]; };

    fn(c.tb, c.M, c.N, c.K, a, lda, c.zp, b, ldb, a_scale, scales.data(), cc, ldc);

    const double u = std::ldexp(1.0, -24), gamma = 2.0 * (c.K + 2) * u;
    double worst = 0.0;
    for (int i = 0; i < c.M; ++i)
        for (int j = 0; j < c.N; ++j) {
            long long sum = 0;
            double mag = 0.0;
            for (int p = 0; p < c.K; ++p) {
                const long long x = (static_cast<long long>(a[(std::size_t)i * lda + p]) - c.zp) * b_at(p, j);
                sum += x;
                mag += std::fabs(static_cast<double>(x));
            }
            const double sc = static_cast<double>(a_scale) * scales[j];
            const double ref = sc * static_cast<double>(sum);
            const double bound = gamma * std::fabs(sc) * mag + 1e-30;
            const double got = cc[(std::size_t)i * ldc + j];
            worst = std::max(worst, std::isfinite(got) ? std::fabs(got - ref) / bound : 1e30);
        }
    for (int i = 0; i < c.M; ++i)
        for (int j = c.N; j < ldc; ++j)
            if (cc[(std::size_t)i * ldc + j] != sentinel) return 1e30;
    for (int j = 0; j < 16; ++j)
        if (cc[(std::size_t)c.M * ldc + j] != sentinel) return 1e30;
    return worst;
}

static bool check(U8S8Fn fn, const char* path, const char* what, const Case& c) {
    const double w = run_case(fn, c);
    const bool pass = w <= 1.0;
    printf("  %-12s %-7s %5dx%5dx%5d %c zp=%3d pad=%d/%d/%d off=%d%s  err/bound=%-8.3g %s\n", path, what,
           c.M, c.N, c.K, c.tb ? 'T' : 'N', c.zp, c.pad_a, c.pad_b, c.pad_c, c.offset,
           c.fill ? " extreme" : "", w, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_path(U8S8Fn fn, const char* path, int kc8, int mc, int nc) {
    bool ok = true;
    for (int s : {1, 2, 3, 4, 5, 7, 8, 9, 13, 64, 97, 128})
        ok &= check(fn, path, "square", {s, s, s});
    for (Case c : {Case{5, 15, 9}, Case{6, 16, 12}, Case{7, 17, 13}, Case{11, 31, 33}, Case{12, 32, 36},
                   Case{13, 33, 35}, Case{25, 65, 70}, Case{1, 4096, 300}, Case{300, 1, 300}, Case{300, 300, 1}})
        ok &= check(fn, path, "shape", c);
    for (bool tb : {false, true}) {
        for (int K : {kc8 - 3, kc8 - 1, kc8, kc8 + 1, kc8 + 2, 2 * kc8 + 5}) {
            Case c{29, 45, K};
            c.tb = tb;
            ok &= check(fn, path, "kc", c);
        }
        for (int M : {mc - 1, mc + 1}) {
            Case c{M, 50, 40};
            c.tb = tb;
            ok &= check(fn, path, "mc", c);
        }
        Case n{9, nc + 1, 21};
        n.tb = tb;
        ok &= check(fn, path, "nc", n);
        for (int zp : {0, 128, 255}) {
            Case z{37, 53, 71};
            z.tb = tb; z.zp = zp;
            ok &= check(fn, path, "zp", z);
        }
        for (int off : {0, 1}) {
            Case st{23, 37, 29, tb, 77, 3, 5, 7, off};
            ok &= check(fn, path, "stride", st);
            Case big{190, 210, 230, tb, 128, 1, 16, 9, off};
            ok &= check(fn, path, "stride", big);
        }
        Case ext{13, 40, kc8, tb, 0};
        ext.fill = 1;
        ok &= check(fn, path, "extreme", ext);
    }
    // One row with B [N, K] (the row-streaming path, int8_rows.hpp): row
    // groups of 4 and 8 ± 1, K tails of the 32-byte step, K across the 2^18
    // int32 chunk, padded strides and odd base pointers.
    for (int N : {1, 3, 4, 5, 7, 8, 9, 17, 4097})
        ok &= check(fn, path, "row", Case{1, N, 64, true});
    for (int K : {1, 31, 32, 33, 63, 65, 300, 4096})
        ok &= check(fn, path, "row", Case{1, 11, K, true});
    for (int zp : {0, 255})
        for (int off : {0, 1}) {
            Case st{1, 13, 77, true, zp, 3, 5, 0, off};
            ok &= check(fn, path, "row", st);
        }
    for (int K : {(1 << 18) - 3, (1 << 18) + 77}) {
        Case big{1, 5, K, true, 0};
        big.fill = 1;   // A = 255, B = -128: every int32 lane near its chunk maximum
        ok &= check(fn, path, "row", big);
        ok &= check(fn, path, "row", Case{1, 3, K, true, 131});
    }
    // K = 0: C = 0 (a sum over nothing); M = 0 / N = 0: nothing written.
    std::vector<float> C(7 * 9, 3.0f), s9(9, 1.0f);   // b_scales: one per column, as always
    const std::uint8_t a1 = 1; const std::int8_t b1 = 1; const float s1 = 1.0f;
    fn(false, 7, 9, 0, &a1, 1, 0, &b1, 9, 1.0f, s9.data(), C.data(), 9);
    bool zero = std::all_of(C.begin(), C.end(), [](float v) { return v == 0.0f; });
    float sentinel = 5.0f;
    fn(false, 0, 1, 1, &a1, 1, 0, &b1, 1, 1.0f, &s1, &sentinel, 1);
    fn(false, 1, 0, 1, &a1, 1, 0, &b1, 1, 1.0f, &s1, &sentinel, 1);
    printf("  %-12s K=0 writes 0, M=0/N=0 write nothing: %s\n", path, zero && sentinel == 5.0f ? "PASS" : "FAIL");
    return ok && zero && sentinel == 5.0f;
}

int run_int8_gemm_tests() {
    printf("\n── INT8 GEMM Tests (KernelTable::sgemm_u8s8) ──\n");
    printf("  avx_vnni_available()=%d  avx512vnni_available()=%d  active path: %s\n",
           simd_ml::gemm::avx_vnni_available(), simd_ml::gemm::avx512vnni_available(),
           simd_ml::gemm::u8s8_path());
    bool ok = for_each_kernel_table([](const simd_ml::KernelTable& t) {
        int kc = 64, mc = 64, nc = 128;
        if (t.sgemm_blocking) {
            int k = 0, m = 0, n = 0;
            t.sgemm_blocking(&k, &m, &n);
            if (k > 0) { kc = k; mc = m; nc = n; }
        }
        return test_path(t.sgemm_u8s8, t.name, 4 * kc, mc, nc);
    });
#if SIMD_ML_X86_AVAILABLE
    // The paths on their own (the table entries pick one of them).
    if (const simd_ml::KernelTable* t = simd_ml::kernels_for(simd_ml::Isa::avx2)) {
        int kc = 0, mc = 0, nc = 0;
        t->sgemm_blocking(&kc, &mc, &nc);
        ok &= test_path(&simd_ml::gemm::sgemm_u8s8_convert_avx2, "fp32_convert", 4 * kc, mc, nc);
#  ifdef SIMD_ML_HAVE_AVXVNNI
        if (simd_ml::gemm::avx_vnni_available())
            ok &= test_path(&simd_ml::gemm::sgemm_u8s8_avxvnni, "avx_vnni", 4 * kc, mc, nc);
        else
            printf("  avx_vnni: SKIP (CPU has no AVX-VNNI)\n");
#  endif
    }
#  ifdef SIMD_ML_HAVE_AVX512VNNI
    if (const simd_ml::KernelTable* t = simd_ml::kernels_for(simd_ml::Isa::avx512);
        t && simd_ml::gemm::avx512vnni_available()) {
        int kc = 0, mc = 0, nc = 0;
        t->sgemm_blocking(&kc, &mc, &nc);
        ok &= test_path(&simd_ml::gemm::sgemm_u8s8_avx512vnni, "avx512_vnni", 4 * kc, mc, nc);
    } else {
        printf("  avx512_vnni: SKIP (CPU has no AVX512_VNNI)\n");
    }
#  endif
#endif
    printf("  Overall: %s\n", ok ? "ALL PASS" : "SOME FAILURES");
    return ok ? 0 : 1;
}
