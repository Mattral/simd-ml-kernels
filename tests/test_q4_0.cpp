/**
 * test_q4_0.cpp — Q4_0 quantization and GEMV (quant/q4_0.hpp).
 *
 *  - quantize_q4_0_rows against ggml's quantize_row_q4_0_ref written out here
 *    (IEEE arithmetic, each step rounded on its own), byte for byte: random
 *    rows, values that land exactly on the truncation points, zero blocks
 *    (scale -0.0), ties of largest magnitude with opposite signs, float
 *    subnormals, binary16-subnormal scales; NaN/±Inf and scale overflow
 *    rejected; 1/d overflow stores q = 8.
 *  - KernelTable::gemv_q4_0 of every table and each inner path called
 *    directly (AVX2 vpmaddubsw, AVX-VNNI, AVX512_VNNI where the CPU has it),
 *    against the float64 sum of the same quantized operands:
 *      |y − y64| <= 2·(K/32 + 4)·u·Σ_b |dw_b·dx_b|·Σ_i |(qw − 8)·qx|,  u = 2^-24;
 *    N across the 8-row groups, K = 0, 32 .. 4128, Wq at an odd address,
 *    extreme quants (0 and 15 against ±127); x with NaN or ±Inf gives NaN
 *    everywhere (bit test).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/kernels/quant/q4_0.hpp"
#include "../src/kernels/gemm/int8_gemm.hpp"
#include "kernel_tables.hpp"

namespace q = simd_ml::quant;

namespace {

unsigned next(unsigned& s) { s = s * 1664525u + 1013904223u; return s >> 8; }
float uniform(unsigned& s) { return static_cast<float>(next(s)) / 8388608.0f - 1.0f; }   // [-1, 1)

std::uint32_t bits(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
float from_bits(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// One rounded binary32 step: the volatile store keeps the compiler from
// fusing w·id + 8.5 into an FMA (ggml's reference rounds twice).
float rounded(float v) { volatile float t = v; return t; }

// ggml's quantize_row_q4_0_ref, with this file's (IEEE) arithmetic.
std::vector<std::uint8_t> ref_quantize(const std::vector<float>& w, int N, int K) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(N) * q::q4_0_row_bytes(K));
    std::uint8_t* o = out.data();
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < K / 32; ++b, o += 18) {
            const float* x = &w[static_cast<std::size_t>(n) * K + 32 * b];
            float amax = 0.0f, max = 0.0f;
            for (int j = 0; j < 32; ++j)
                if (amax < std::fabs(x[j])) { amax = std::fabs(x[j]); max = x[j]; }
            const float d = max / -8;
            float id = d ? 1.0f / d : 0.0f;
            if (std::isinf(id)) id = 0.0f;   // ggml: undefined float -> int8 conversion
            const std::uint16_t h = q::fp32_to_fp16(d);
            o[0] = static_cast<std::uint8_t>(h & 0xff);
            o[1] = static_cast<std::uint8_t>(h >> 8);
            for (int j = 0; j < 16; ++j) {
                const float x0 = rounded(x[j] * id), x1 = rounded(x[16 + j] * id);
                const int xi0 = std::min(15, static_cast<int>(static_cast<std::int8_t>(rounded(x0 + 8.5f))));
                const int xi1 = std::min(15, static_cast<int>(static_cast<std::int8_t>(rounded(x1 + 8.5f))));
                o[2 + j] = static_cast<std::uint8_t>(xi0 | (xi1 << 4));
            }
        }
    return out;
}

bool quantize_case(const char* what, const std::vector<float>& w, int N, int K) {
    std::vector<std::uint8_t> got(static_cast<std::size_t>(N) * q::q4_0_row_bytes(K) + 1, 0xEE);
    const bool ret = q::quantize_q4_0_rows(w.data(), N, K, got.data());
    const std::vector<std::uint8_t> want = ref_quantize(w, N, K);
    const bool same = std::memcmp(got.data(), want.data(), want.size()) == 0 && got.back() == 0xEE;
    printf("  quantize %-30s %4dx%5d: %s\n", what, N, K, ret && same ? "PASS" : "FAIL");
    for (std::size_t i = 0; i < want.size(); ++i)
        if (got[i] != want[i]) {
            printf("    first difference: byte %zu (block %zu) got 0x%02x want 0x%02x\n", i, i / 18, got[i], want[i]);
            break;
        }
    return ret && same;
}

bool test_quantize() {
    bool ok = true;
    unsigned s = 7;
    for (auto [N, K] : {std::pair{1, 32}, {3, 64}, {17, 4096}, {64, 352}}) {
        std::vector<float> w(static_cast<std::size_t>(N) * K);
        for (float& v : w) v = uniform(s) * 3.0f;
        ok &= quantize_case("random", w, N, K);
    }
    // max = -8 (d = 1, id = 1): w + 8.5 lands on k + 0.5 and on integers;
    // the second block has +8 as its largest magnitude (d = -1).
    std::vector<float> steps(96);
    for (int j = 0; j < 32; ++j) steps[j] = -8.0f + 0.5f * j;   // -8 .. 7.5
    for (int j = 0; j < 32; ++j) steps[32 + j] = 8.0f - 0.5f * j;
    // A zero block (d = -0.0, every q = 8) at j 64..95.
    for (int j = 64; j < 96; ++j) steps[j] = 0.0f;
    ok &= quantize_case("steps + zero block", steps, 1, 96);
    // Largest magnitude twice with opposite signs: the first one sets the sign.
    std::vector<float> tie(64);
    for (int j = 0; j < 64; ++j) tie[j] = 0.25f * uniform(s);
    tie[5] = 2.0f; tie[9] = -2.0f; tie[32 + 3] = -2.0f; tie[32 + 30] = 2.0f;
    ok &= quantize_case("opposite-sign ties", tie, 1, 64);
    std::vector<float> tiny(96);
    for (int j = 0; j < 32; ++j) tiny[j] = from_bits(0x00800000u + 977u * j) * ((j & 1) ? -1.0f : 1.0f);  // ~1.2e-38
    for (int j = 32; j < 64; ++j) tiny[j] = 1e-5f * uniform(s);    // d a binary16 subnormal
    for (int j = 64; j < 96; ++j) tiny[j] = 5.2e5f * uniform(s);   // large, below the overflow
    ok &= quantize_case("subnormal / large scales", tiny, 1, 96);
    std::vector<std::uint8_t> out(18 * 2);
    // |max| < 8/FLT_MAX: 1/d overflows; q = 8 (value 0), scale +-0.
    std::vector<float> under(32);
    for (int j = 0; j < 32; ++j) under[j] = from_bits(0x00000001u + 3u * j) * ((j & 1) ? -1.0f : 1.0f);
    bool rejects = q::quantize_q4_0_rows(under.data(), 1, 32, out.data());
    rejects &= (out[0] | (out[1] & 0x7f)) == 0;
    for (int j = 0; j < 16; ++j) rejects &= out[2 + j] == 0x88;
    std::vector<float> bad(64, 1.0f);
    for (std::uint32_t pat : {0x7fc00000u, 0x7f800000u, 0xff800000u}) {
        bad[40] = from_bits(pat);
        rejects &= !q::quantize_q4_0_rows(bad.data(), 1, 64, out.data());
    }
    bad[40] = 1.0f;
    bad[3] = 8.0f * 65520.0f;   // d rounds to binary16 infinity
    rejects &= !q::quantize_q4_0_rows(bad.data(), 1, 64, out.data());
    bad[3] = -8.0f * 65504.0f;  // the largest scale: accepted
    rejects &= q::quantize_q4_0_rows(bad.data(), 1, 64, out.data());
    printf("  quantize: 1/d overflow -> q = 8; rejects NaN/±Inf and binary16 scale overflow: %s\n",
           rejects ? "PASS" : "FAIL");
    return ok && rejects;
}

// Worst |err| / bound over y; 1e30 if a value is not finite or y's guard changed.
double gemv_case(simd_ml::GemvQ4_0Fn fn, int N, int K, int offset, int fill) {
    unsigned s = 99u + static_cast<unsigned>(N * 131 + K * 7 + offset);
    std::vector<float> w(static_cast<std::size_t>(N) * K), x(static_cast<std::size_t>(K) + 1);
    // fill: weights -8 or +7 times a block's |max| (q = 0 or 15; d < 0 when
    // the block's first largest value is -8), x ±2 (every quant ±127).
    for (std::size_t i = 0; i < w.size(); ++i)
        w[i] = fill ? ((i % 32 == 0 || (next(s) & 1)) ? -8.0f : 7.0f) : uniform(s) * (1 + next(s) % 4);
    for (int k = 0; k < K; ++k) x[k] = fill ? ((next(s) & 1) ? 2.0f : -2.0f) : uniform(s) * 5.0f;
    const std::size_t bytes = static_cast<std::size_t>(N) * q::q4_0_row_bytes(K);
    std::vector<std::uint8_t> store(bytes + offset + 1);
    std::uint8_t* wq = store.data() + offset;
    if (!q::quantize_q4_0_rows(w.data(), N, K, wq)) return 1e30;
    std::vector<std::int8_t> xq(static_cast<std::size_t>(K) + 1);
    std::vector<float> dx(static_cast<std::size_t>(K / 32) + 1);
    if (!q::quantize_x_q8(x.data(), K, xq.data(), dx.data())) return 1e30;
    const float guard = -3.5f;
    std::vector<float> y(static_cast<std::size_t>(N) + 1, guard);
    fn(N, K, x.data(), wq, y.data());
    const double u = std::ldexp(1.0, -24), gamma = 2.0 * (K / 32 + 4) * u;
    double worst = 0.0;
    for (int n = 0; n < N; ++n) {
        double ref = 0.0, mag = 0.0;
        for (int b = 0; b < K / 32; ++b) {
            const std::uint8_t* blk = wq + n * q::q4_0_row_bytes(K) + 18 * b;
            const double sc = static_cast<double>(q::fp16_to_fp32(static_cast<std::uint16_t>(blk[0] | blk[1] << 8))) * dx[b];
            long long isum = 0, amag = 0;
            for (int i = 0; i < 32; ++i) {
                const int nib = i < 16 ? (blk[2 + i] & 0x0f) : (blk[2 + i - 16] >> 4);
                const long long p = static_cast<long long>(nib - 8) * xq[32 * b + i];
                isum += p;
                amag += p < 0 ? -p : p;
            }
            ref += sc * static_cast<double>(isum);
            mag += std::fabs(sc) * static_cast<double>(amag);
        }
        const double got = y[n];
        worst = std::max(worst, std::isfinite(got) ? std::fabs(got - ref) / (gamma * mag + 1e-300) : 1e30);
    }
    return y[N] == guard ? worst : 1e30;
}

bool test_gemv(simd_ml::GemvQ4_0Fn fn, const char* path) {
    bool ok = true;
    auto run = [&](const char* what, int N, int K, int off, int fill) {
        const double w = gemv_case(fn, N, K, off, fill);
        const bool pass = w <= 1.0;
        printf("  %-12s %-7s N=%5d K=%5d off=%d  err/bound=%-8.3g %s\n", path, what, N, K, off, w, pass ? "PASS" : "FAIL");
        ok &= pass;
    };
    for (int N : {1, 2, 7, 8, 9, 15, 16, 17, 31, 100}) run("rows", N, 64, 0, 0);
    for (int K : {32, 96, 128, 160, 256, 1024, 4128}) run("k", 13, K, 0, 0);
    run("odd", 11, 160, 1, 0);
    run("extreme", 9, 4096, 3, 1);
    // K = 0: y = 0; N = 0: nothing written; x with NaN / ±Inf: y all NaN.
    std::vector<float> y(5, 7.0f);
    const float x0 = 1.0f;
    const std::uint8_t w0 = 0;
    fn(5, 0, &x0, &w0, y.data());
    bool special = std::all_of(y.begin(), y.end(), [](float v) { return v == 0.0f; });
    float untouched = 7.0f;
    fn(0, 32, &x0, &w0, &untouched);
    special &= untouched == 7.0f;
    std::vector<float> w(3 * 64, 0.25f), x(64, 1.0f);
    std::vector<std::uint8_t> wq(3 * q::q4_0_row_bytes(64));
    q::quantize_q4_0_rows(w.data(), 3, 64, wq.data());
    for (std::uint32_t pat : {0x7fc00000u, 0x7f800000u, 0xff800000u}) {
        x[45] = from_bits(pat);
        fn(3, 64, x.data(), wq.data(), y.data());
        for (int n = 0; n < 3; ++n) special &= (bits(y[n]) & 0x7fffffffu) > 0x7f800000u;
    }
    printf("  %-12s K=0 -> 0, N=0 writes nothing, NaN/±Inf in x -> NaN: %s\n", path, special ? "PASS" : "FAIL");
    return ok && special;
}

// The inner paths on their own, through the shared driver.
template <q::Q8RowsFn Rows>
void via_driver(int N, int K, const float* x, const std::uint8_t* Wq, float* y) {
    q::gemv_q8_0_run(N, K, x, Wq, y, Rows, q::kQ4_0Bytes);
}

}  // namespace

int run_q4_0_tests() {
    printf("\n── Q4_0 Tests (quantize_q4_0_rows, KernelTable::gemv_q4_0) ──\n");
    printf("  active path: %s\n", q::q8_0_path());
    bool ok = test_quantize();
    ok &= for_each_kernel_table([](const simd_ml::KernelTable& t) { return test_gemv(t.gemv_q4_0, t.name); });
#if SIMD_ML_X86_AVAILABLE
    if (simd_ml::kernels_for(simd_ml::Isa::avx2)) {
        ok &= test_gemv(&via_driver<&q::q4_0_rows_avx2>, "avx2");
#  ifdef SIMD_ML_HAVE_AVXVNNI
        if (simd_ml::gemm::avx_vnni_available()) ok &= test_gemv(&via_driver<&q::q4_0_rows_avxvnni>, "avx_vnni");
        else printf("  avx_vnni: SKIP (CPU has no AVX-VNNI)\n");
#  endif
    }
#  ifdef SIMD_ML_HAVE_AVX512VNNI
    if (simd_ml::kernels_for(simd_ml::Isa::avx512) && simd_ml::gemm::avx512vnni_available())
        ok &= test_gemv(&via_driver<&q::q4_0_rows_avx512vnni>, "avx512_vnni");
    else
        printf("  avx512_vnni: SKIP (CPU has no AVX512_VNNI)\n");
#  endif
#endif
    printf("  Overall: %s\n", ok ? "ALL PASS" : "SOME FAILURES");
    return ok ? 0 : 1;
}
