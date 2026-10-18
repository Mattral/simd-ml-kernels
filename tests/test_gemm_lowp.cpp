/**
 * test_gemm_lowp.cpp — FP16- and BF16-input GEMM (KernelTable::sgemm_f16 /
 * sgemm_bf16) of every table the CPU supports, against a float64 reference.
 *
 * Both conversions to FP32 are exact, so the reference is the float64 product
 * of the converted inputs and the bound is the FP32 GEMM's (test_gemm_packed.cpp):
 * |C − C64| ≤ γ·(|α|·(|A||B|)ij + |β|·|C0ij|), γ = 2·(K+2)·u, u = 2^-24.
 *
 * Shapes come from the table's FP32 blocking (the low-precision GEMMs use the
 * FP32 loops): K, M, N just past KC, MC, NC; M, N around the 6×16 / 12×32
 * tiles; the direct path's shapes (M = 1, N <= 16, products up to 80³, and its
 * convert-B-once case M > 6). Strided cases pad A, B and C with NaN bit
 * patterns, and pointers are offset by one element (2-byte alignment).
 * Special values: a NaN in A or B makes exactly its row / column of C NaN;
 * FP16 and BF16 subnormals are converted, not flushed (no DAZ in the kernels).
 */

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>

#include "../src/kernels/gemm/avx2_gemm_packed.hpp"
#include "../src/kernels/gemm/bf16_gemm.hpp"
#include "../src/kernels/gemm/f16_gemm.hpp"
#include "kernel_tables.hpp"

// The table the test functions below call into (set by the runner).
static const simd_ml::KernelTable* T = nullptr;

// ─── Conversions in software ─────────────────────────────────────────────────
// The test binary is built for baseline x86-64 (no F16C), like the scalar table.

static float bits_f32(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t f32_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

/** FP32 -> FP16, round to nearest even; finite |f| < 65504. */
static uint16_t f32_to_f16(float f) {
    const uint32_t u = f32_bits(f);
    const uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000u);
    const float a = std::fabs(f);
    if (a < 6.103515625e-05f)   // FP16 subnormal (< 2^-14) or zero, in steps of 2^-24
        return static_cast<uint16_t>(sign | static_cast<uint16_t>(std::nearbyint(a * 16777216.0f)));
    const uint32_t man = u & 0x7fffffu, exp = ((u >> 23) & 0xffu) - 127 + 15;
    uint32_t h = (exp << 10) | (man >> 13);
    const uint32_t rest = man & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return static_cast<uint16_t>(sign | h);
}

static float f16_to_f32(uint16_t h) {
    const float sign = (h & 0x8000u) ? -1.0f : 1.0f;
    const int exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    if (exp == 0) return sign * std::ldexp(static_cast<float>(man), -24);
    if (exp == 31) return man ? std::nanf("") : sign * INFINITY;
    return sign * bits_f32(static_cast<uint32_t>(exp - 15 + 127) << 23 | static_cast<uint32_t>(man) << 13);
}

/** FP32 -> BF16, round to nearest even (finite inputs). */
static uint16_t f32_to_bf16(float f) {
    const uint32_t u = f32_bits(f);
    return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

static float bf16_to_f32(uint16_t b) { return bits_f32(static_cast<uint32_t>(b) << 16); }

struct Fmt {
    const char* name;
    uint16_t (*enc)(float);
    float (*dec)(uint16_t);
    uint16_t nan;
    simd_ml::Sgemm16Fn simd_ml::KernelTable::*fn;
};
static const Fmt kF16{"f16", f32_to_f16, f16_to_f32, 0x7e00, &simd_ml::KernelTable::sgemm_f16};
static const Fmt kBF16{"bf16", f32_to_bf16, bf16_to_f32, 0x7fc0, &simd_ml::KernelTable::sgemm_bf16};

static void fill_random(float* buf, std::size_t n, unsigned seed) {
    unsigned s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        buf[i] = static_cast<float>(int(s >> 16) % 2000 - 1000) * 1e-3f;
    }
}

struct Case {
    int M, N, K;
    float alpha = 1.0f, beta = 0.0f;
    int pad_a = 0, pad_b = 0, pad_c = 0;  // extra elements per row (lda = K + pad_a, ...)
    int offset = 0;                       // elements added to every base pointer
};

// Runs the case; returns the worst |err| / bound (pass: <= 1), or 1e30 if a
// padding sentinel of C changed.
static double run_case(const Fmt& f, const Case& c) {
    const int lda = c.K + c.pad_a, ldb = c.N + c.pad_b, ldc = c.N + c.pad_c;
    const float nan = std::nanf("");
    std::vector<uint16_t> A(c.offset + (std::size_t)c.M * lda + 1, f.nan),
        B(c.offset + (std::size_t)c.K * ldb + 1, f.nan);
    std::vector<float> C(c.offset + (std::size_t)c.M * ldc + 16, nan);
    uint16_t* a = A.data() + c.offset; uint16_t* b = B.data() + c.offset; float* cc = C.data() + c.offset;
    std::vector<float> va((std::size_t)c.M * c.K), vb((std::size_t)c.K * c.N);
    fill_random(va.data(), va.size(), 1);
    fill_random(vb.data(), vb.size(), 2);
    std::vector<double> ra(va.size()), rb(vb.size());   // the values the kernel sees
    for (int i = 0; i < c.M; ++i)
        for (int p = 0; p < c.K; ++p) {
            const uint16_t h = f.enc(va[(std::size_t)i * c.K + p]);
            a[(std::size_t)i * lda + p] = h;
            ra[(std::size_t)i * c.K + p] = f.dec(h);
        }
    for (int p = 0; p < c.K; ++p)
        for (int j = 0; j < c.N; ++j) {
            const uint16_t h = f.enc(vb[(std::size_t)p * c.N + j]);
            b[(std::size_t)p * ldb + j] = h;
            rb[(std::size_t)p * c.N + j] = f.dec(h);
        }
    std::vector<float> c0((std::size_t)c.M * c.N);
    fill_random(c0.data(), c0.size(), 3);
    if (c.beta == 0.0f) std::fill(c0.begin(), c0.end(), nan);   // beta = 0 must not read C
    for (int i = 0; i < c.M; ++i) std::copy_n(&c0[(std::size_t)i * c.N], c.N, cc + (std::size_t)i * ldc);

    (T->*f.fn)(c.M, c.N, c.K, c.alpha, a, lda, b, ldb, c.beta, cc, ldc);

    const double gamma = 2.0 * (c.K + 2) * std::ldexp(1.0, -24);
    double worst = 0.0;
    for (int i = 0; i < c.M; ++i)
        for (int j = 0; j < c.N; ++j) {
            double ref = 0.0, mag = 0.0;
            for (int p = 0; p < c.K; ++p) {
                const double x = ra[(std::size_t)i * c.K + p] * rb[(std::size_t)p * c.N + j];
                ref += x; mag += std::fabs(x);
            }
            const double cin = c.beta == 0.0f ? 0.0 : c0[(std::size_t)i * c.N + j];
            ref = c.alpha * ref + c.beta * cin;
            const double bound = gamma * (std::fabs(c.alpha) * mag + std::fabs(c.beta * cin)) + 1e-30;
            const double got = cc[(std::size_t)i * ldc + j];
            worst = std::max(worst, std::isnan(got) ? 1e30 : std::fabs(got - ref) / bound);
        }
    for (int i = 0; i < c.M; ++i)
        for (int j = c.N; j < ldc; ++j)
            if (!std::isnan(cc[(std::size_t)i * ldc + j])) return 1e30;
    for (int j = 0; j < 16; ++j)
        if (!std::isnan(cc[(std::size_t)c.M * ldc + j])) return 1e30;
    return worst;
}

static bool check(const Fmt& f, const char* what, const Case& c) {
    const double w = run_case(f, c);
    const bool pass = w <= 1.0;
    printf("  %-4s %-7s %5dx%5dx%5d a=%.1f b=%.1f pad=%d/%d/%d off=%d  err/bound=%-8.3g %s\n",
           f.name, what, c.M, c.N, c.K, c.alpha, c.beta, c.pad_a, c.pad_b, c.pad_c, c.offset, w,
           pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_shapes(const Fmt& f) {
    int kc = 0, mc = 0, nc = 0;
    T->sgemm_blocking(&kc, &mc, &nc);
    bool ok = true;
    for (int s : {1, 7, 8, 9, 13, 64, 97, 128, 129, 256})
        ok &= check(f, "square", {s, s, s});
    for (Case c : {Case{5, 15, 9}, Case{6, 16, 9}, Case{7, 17, 9}, Case{11, 31, 33}, Case{12, 32, 33},
                   Case{13, 33, 33}, Case{25, 65, 70}})
        ok &= check(f, "tile", c);
    // Direct path: M = 1, N <= 16, small products, M > 6 (B converted once).
    for (Case c : {Case{1, 4096, 300}, Case{1, 33, 70}, Case{4, 64, 256}, Case{300, 16, 300},
                   Case{300, 7, 17}, Case{6, 300, 40}, Case{7, 300, 40}, Case{50, 60, 70}})
        ok &= check(f, "direct", c);
    if (kc > 0) {
        for (int K : {kc - 1, kc, kc + 1, 2 * kc + 3}) ok &= check(f, "kc", {97, 113, K});
        for (int M : {mc - 1, mc, mc + 1}) ok &= check(f, "mc", {M, 65, 33});
        ok &= check(f, "nc", {29, nc + 1, 19});
        ok &= check(f, "nc", {29, 2 * nc + 1, 19});
    }
    for (float alpha : {1.0f, 2.5f, -0.5f})
        for (float beta : {0.0f, 1.0f, 0.3f}) {
            ok &= check(f, "ab", {33, 41, 27, alpha, beta});
            ok &= check(f, "ab", {200, 190, 180, alpha, beta});
        }
    for (int off : {0, 1}) {
        ok &= check(f, "stride", {23, 37, 29, 1.0f, 0.5f, 3, 5, 7, off});
        ok &= check(f, "stride", {190, 210, 230, 1.0f, 0.0f, 1, 16, 9, off});
        ok &= check(f, "stride", {1, 210, 230, 1.0f, 0.0f, 1, 16, 9, off});
    }
    return ok;
}

// K == 0: C = beta·C, beta = 0 overwrites NaN; M or N == 0: nothing written.
static bool test_degenerate(const Fmt& f) {
    bool ok = true;
    for (int M : {7, 300}) {
        const int N = M == 7 ? 9 : 400;
        for (float beta : {0.0f, 0.5f}) {
            std::vector<float> C((std::size_t)M * N, beta == 0.0f ? std::nanf("") : 3.0f);
            const uint16_t dummy = 0x3c00;
            (T->*f.fn)(M, N, 0, 1.0f, &dummy, 1, &dummy, N, beta, C.data(), N);
            const float expect = beta == 0.0f ? 0.0f : beta * 3.0f;
            bool pass = true;
            for (float v : C) pass &= v == expect;
            printf("  %-4s K=0 M=%-3d beta=%.1f: %s\n", f.name, M, beta, pass ? "PASS" : "FAIL");
            ok &= pass;
        }
    }
    float sentinel = 7.0f;
    const uint16_t one = f.enc(1.0f);
    (T->*f.fn)(0, 5, 3, 1.0f, &one, 3, &one, 5, 0.0f, &sentinel, 5);
    (T->*f.fn)(5, 0, 3, 1.0f, &one, 3, &one, 1, 0.0f, &sentinel, 1);
    printf("  %-4s M=0 / N=0 write nothing: %s\n", f.name, sentinel == 7.0f ? "PASS" : "FAIL");
    return ok && sentinel == 7.0f;
}

// One NaN in A(i0, p0) and one in B(p1, j0): exactly row i0 and column j0 of C
// are NaN, on the direct and on the packed path.
static bool test_nan(const Fmt& f) {
    bool ok = true;
    for (int n : {40, 200}) {
        const int M = n, N = n + 3, K = n - 1, i0 = 5, j0 = N - 2;
        std::vector<uint16_t> A((std::size_t)M * K, f.enc(0.25f)), B((std::size_t)K * N, f.enc(-0.5f));
        A[(std::size_t)i0 * K + 3] = f.nan;
        B[(std::size_t)(K - 1) * N + j0] = f.nan;
        std::vector<float> C((std::size_t)M * N, 0.0f);
        (T->*f.fn)(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
        bool pass = true;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                const float v = C[(std::size_t)i * N + j];
                pass &= (i == i0 || j == j0) ? std::isnan(v) : v == -0.125f * K;
            }
        printf("  %-4s NaN row/column propagation %dx%dx%d: %s\n", f.name, M, N, K, pass ? "PASS" : "FAIL");
        ok &= pass;
    }
    return ok;
}

// Whether the AVX-512 table's BF16 GEMM takes the vdpbf16ps path for this
// shape (avx512bf16_gemm.cpp; built by every compiler CI and the dev builds
// use): opted in, AVX512_BF16 hardware (or SDE -spr), a packed shape, one thread.
static bool uses_vdpbf16ps(const Fmt& f, int M, int N, int K) {
#if SIMD_ML_X86_AVAILABLE   // sgemm_avx2_prefers_direct is in the AVX2 unit
    return f.fn == kBF16.fn && T->isa == simd_ml::Isa::avx512 && simd_ml::gemm::bf16_dot_enabled() &&
           !simd_ml::gemm::sgemm_avx2_prefers_direct(M, N, K) &&
           simd_ml::gemm::gemm_threads(M, N, K, simd_ml::gemm::MR512, simd_ml::gemm::NR512) == 1;
#else
    (void)f; (void)M; (void)N; (void)K;
    return false;
#endif
}

// Subnormal inputs: A holds the smallest subnormals of the format, B a power
// of two that brings the products into the normal FP32 range; any flush to
// zero (DAZ) would give 0. vdpbf16ps reads BF16 subnormals as zero by
// definition (it ignores MXCSR), so on that path the expected result is 0.
static bool test_subnormals(const Fmt& f) {
    // FP16: 0x0001 = 2^-24, 0x03ff = 1023·2^-24; times 2^10 (0x6400).
    // BF16: 0x0001 = 2^-133, 0x007f = 127·2^-133; times 2^100 (0x7180).
    const bool is_f16 = f.fn == kF16.fn;
    const uint16_t sub[2] = {0x0001, static_cast<uint16_t>(is_f16 ? 0x03ff : 0x007f)};
    const uint16_t scale = is_f16 ? 0x6400 : 0x7180;
    bool ok = true;
    for (int n : {8, 200}) {   // direct and packed paths
        const int M = n, N = n, K = n;
        std::vector<uint16_t> A((std::size_t)M * K, 0), B((std::size_t)K * N, 0);
        for (int i = 0; i < M; ++i) A[(std::size_t)i * K + i % K] = sub[i & 1];
        for (int p = 0; p < K; ++p) B[(std::size_t)p * N + p % N] = scale;   // diagonal
        std::vector<float> C((std::size_t)M * N, 0.0f);
        (T->*f.fn)(M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N);
        const bool daz = uses_vdpbf16ps(f, M, N, K);
        bool pass = true;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j) {
                const double exact = j == i % K ? (double)f.dec(sub[i & 1]) * f.dec(scale) : 0.0;
                pass &= C[(std::size_t)i * N + j] == (daz ? 0.0f : static_cast<float>(exact)) &&
                        (daz || j != i % K || exact != 0.0);
            }
        printf("  %-4s subnormal inputs %s (n=%d): %s\n", f.name,
               daz ? "read as zero (vdpbf16ps)" : "not flushed", n, pass ? "PASS" : "FAIL");
        ok &= pass;
    }
    return ok;
}

int run_lowp_gemm_tests() {
    printf("\n── FP16 / BF16 GEMM Tests (KernelTable::sgemm_f16 / sgemm_bf16) ──\n");
    printf("  f16c_available()=%d  bf16_avx512bf16_available()=%d\n",
           simd_ml::gemm::f16c_available(), simd_ml::gemm::bf16_avx512bf16_available());
#ifdef SIMD_ML_OPENMP
    const int thread_counts[] = {1, 4};
#else
    const int thread_counts[] = {1};
#endif
    const int saved_threads = simd_ml::gemm::get_num_threads();
    bool all_pass = true;
    for (int threads : thread_counts) {
        simd_ml::gemm::set_num_threads(threads);
        printf(" Threads: %d\n", threads);
        all_pass &= for_each_kernel_table([](const simd_ml::KernelTable& t) {
            T = &t;
            bool ok = true;
            for (const Fmt* f : {&kF16, &kBF16}) {
                ok &= test_shapes(*f);
                ok &= test_degenerate(*f);
                ok &= test_nan(*f);
                ok &= test_subnormals(*f);
            }
            return ok;
        });
    }
    simd_ml::gemm::set_num_threads(saved_threads);
    // The vdpbf16ps path is opt-in (SIMD_ML_BF16_DOT=1), so the loop above
    // covers it only when the environment enables it. Run the BF16 suite on the
    // AVX-512 table once more with it forced on (AVX512_BF16 or SDE -spr).
    if (simd_ml::gemm::bf16_avx512bf16_available()) {
        const bool saved_dot = simd_ml::gemm::bf16_dot_enabled();
        simd_ml::gemm::set_bf16_dot(true);
        simd_ml::gemm::set_num_threads(1);
        printf(" vdpbf16ps path (forced on):\n");
        all_pass &= for_each_kernel_table([](const simd_ml::KernelTable& t) {
            if (t.isa != simd_ml::Isa::avx512) return true;
            T = &t;
            bool ok = test_shapes(kBF16);
            ok &= test_degenerate(kBF16);
            ok &= test_nan(kBF16);
            ok &= test_subnormals(kBF16);
            return ok;
        });
        simd_ml::gemm::set_bf16_dot(saved_dot);
        simd_ml::gemm::set_num_threads(saved_threads);
    }
    printf("  Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILURES");
    return all_pass ? 0 : 1;
}
