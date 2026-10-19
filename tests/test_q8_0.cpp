/**
 * test_q8_0.cpp — Q8_0 quantization and GEMV (quant/q8_0.hpp).
 *
 *  - fp32_to_fp16: every binary16 value round-trips; every midpoint between
 *    two neighbours rounds to the even one, and one float ulp either side of
 *    it to the nearer one; overflow, NaN and ±Inf.
 *  - quantize_q8_0_rows against ggml's quantize_row_q8_0_ref written out
 *    here (libm roundf; this file has no -ffast-math), byte for byte: random
 *    rows, exact .5 ties, zero blocks, float subnormals, scales that are
 *    binary16 subnormals; NaN/±Inf and scale overflow rejected.
 *  - KernelTable::gemv_q8_0 of every table and each inner path called
 *    directly (AVX2 vpmaddubsw, AVX-VNNI, AVX512_VNNI where the CPU has it),
 *    against the float64 sum of the same quantized operands:
 *      |y − y64| <= 2·(K/32 + 4)·u·Σ_b |dw_b·dx_b|·Σ_i |qw·qx|,  u = 2^-24;
 *    N across the 8-row groups, K = 0, 32 .. 4128, Wq at an odd address,
 *    all-±127 quants; x with NaN or ±Inf gives NaN everywhere (bit test).
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/kernels/quant/q8_0.hpp"
#include "../src/kernels/gemm/int8_gemm.hpp"
#include "kernel_tables.hpp"

namespace q = simd_ml::quant;

namespace {

unsigned next(unsigned& s) { s = s * 1664525u + 1013904223u; return s >> 8; }
float uniform(unsigned& s) { return static_cast<float>(next(s)) / 8388608.0f - 1.0f; }   // [-1, 1): next() < 2^24

std::uint32_t bits(float f) { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
float from_bits(std::uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

bool test_fp16() {
    bool ok = true;
    int bad = 0;
    for (std::uint32_t h = 0; h < 0x7c00u; ++h) {   // finite, positive
        const float f = q::fp16_to_fp32(static_cast<std::uint16_t>(h));
        if (q::fp32_to_fp16(f) != h || q::fp32_to_fp16(-f) != (h | 0x8000u)) ++bad;
        if (h + 1 < 0x7c00u) {
            const float g = q::fp16_to_fp32(static_cast<std::uint16_t>(h + 1));
            const float mid = static_cast<float>((static_cast<double>(f) + g) / 2);   // exact in binary32
            const std::uint32_t even = (h & 1u) ? h + 1 : h;
            if (q::fp32_to_fp16(mid) != even) ++bad;
            if (q::fp32_to_fp16(std::nextafter(mid, 0.0f)) != h) ++bad;
            if (q::fp32_to_fp16(std::nextafter(mid, 1e9f)) != h + 1) ++bad;
        }
    }
    // 65504 is the largest finite half; 65520 (the midpoint to 2^16) rounds to infinity.
    bad += q::fp32_to_fp16(65519.996f) != 0x7bffu;
    bad += q::fp32_to_fp16(65520.0f) != 0x7c00u;
    bad += q::fp32_to_fp16(from_bits(0x7f800000u)) != 0x7c00u;
    bad += q::fp32_to_fp16(from_bits(0xff800000u)) != 0xfc00u;
    bad += q::fp32_to_fp16(from_bits(0x7fc00001u)) != 0x7e00u;
    bad += q::fp32_to_fp16(1e-30f) != 0u;            // below half the smallest subnormal
    bad += q::fp32_to_fp16(std::ldexp(1.0f, -25)) != 0u;   // exactly half of 2^-24: ties to even (0)
    bad += q::fp32_to_fp16(std::nextafter(std::ldexp(1.0f, -25), 1.0f)) != 1u;
    ok = bad == 0;
    printf("  fp32_to_fp16 (every half, midpoints, overflow, NaN): %d mismatches %s\n", bad, ok ? "PASS" : "FAIL");
    return ok;
}

// ggml's quantize_row_q8_0_ref, with this file's (IEEE) arithmetic and libm roundf.
std::vector<std::uint8_t> ref_quantize(const std::vector<float>& w, int N, int K) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(N) * q::q8_0_row_bytes(K));
    std::uint8_t* o = out.data();
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < K / 32; ++b, o += 34) {
            const float* x = &w[static_cast<std::size_t>(n) * K + 32 * b];
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) amax = amax > std::fabs(x[j]) ? amax : std::fabs(x[j]);
            const float d = amax / ((1 << 7) - 1);
            float id = d ? 1.0f / d : 0.0f;
            if (std::isinf(id)) id = 0.0f;   // ggml: undefined float -> int8 conversion (0 on x86)
            const std::uint16_t h = q::fp32_to_fp16(d);
            o[0] = static_cast<std::uint8_t>(h & 0xff);
            o[1] = static_cast<std::uint8_t>(h >> 8);
            for (int j = 0; j < 32; ++j) o[2 + j] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::roundf(x[j] * id)));
        }
    return out;
}

bool quantize_case(const char* what, const std::vector<float>& w, int N, int K) {
    std::vector<std::uint8_t> got(static_cast<std::size_t>(N) * q::q8_0_row_bytes(K) + 1, 0xEE);
    const bool ret = q::quantize_q8_0_rows(w.data(), N, K, got.data());
    const std::vector<std::uint8_t> want = ref_quantize(w, N, K);
    const bool same = std::memcmp(got.data(), want.data(), want.size()) == 0 && got.back() == 0xEE;
    printf("  quantize %-28s %4dx%5d: %s\n", what, N, K, ret && same ? "PASS" : "FAIL");
    for (std::size_t i = 0; i < want.size(); ++i)
        if (got[i] != want[i]) {
            printf("    first difference: byte %zu (block %zu) got 0x%02x want 0x%02x\n", i, i / 34, got[i], want[i]);
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
    // amax = 127 (d = 1): k + 0.5 rounds away from zero, as roundf.
    std::vector<float> ties(64);
    for (int j = 0; j < 32; ++j) ties[j] = (j - 16) + 0.5f;
    ties[31] = 127.0f;
    for (int j = 32; j < 64; ++j) ties[j] = 0.0f;   // a zero block: d = 0, q = 0
    ok &= quantize_case("ties + zero block", ties, 1, 64);
    std::vector<float> tiny(96);
    for (int j = 0; j < 32; ++j) tiny[j] = from_bits(0x00000001u + 977u * j) * ((j & 1) ? -1.0f : 1.0f);  // float subnormals
    for (int j = 32; j < 64; ++j) tiny[j] = 1e-4f * uniform(s);    // d a binary16 subnormal
    for (int j = 64; j < 96; ++j) tiny[j] = 8.0e6f * uniform(s);   // large, below the overflow
    ok &= quantize_case("subnormal / large scales", tiny, 1, 96);
    std::vector<std::uint8_t> out(34 * 2);
    std::vector<float> bad(64, 1.0f);
    bool rejects = true;
    for (std::uint32_t pat : {0x7fc00000u, 0x7f800000u, 0xff800000u}) {
        bad[40] = from_bits(pat);
        rejects &= !q::quantize_q8_0_rows(bad.data(), 1, 64, out.data());
    }
    bad[40] = 1.0f;
    bad[3] = 127.0f * 65520.0f;   // d rounds to binary16 infinity
    rejects &= !q::quantize_q8_0_rows(bad.data(), 1, 64, out.data());
    bad[3] = 127.0f * 65504.0f;   // the largest scale: accepted
    rejects &= q::quantize_q8_0_rows(bad.data(), 1, 64, out.data());
    printf("  quantize rejects NaN/±Inf and binary16 scale overflow: %s\n", rejects ? "PASS" : "FAIL");
    return ok && rejects;
}

// Worst |err| / bound over y; 1e30 if a value is not finite or y's guard changed.
double gemv_case(simd_ml::GemvQ8_0Fn fn, int N, int K, int offset, int fill) {
    unsigned s = 99u + static_cast<unsigned>(N * 131 + K * 7 + offset);
    std::vector<float> w(static_cast<std::size_t>(N) * K), x(static_cast<std::size_t>(K) + 1);
    for (float& v : w) v = fill ? ((next(s) & 1) ? 1.0f : -1.0f) : uniform(s) * (1 + next(s) % 4);
    for (int k = 0; k < K; ++k) x[k] = fill ? ((next(s) & 1) ? 2.0f : -2.0f) : uniform(s) * 5.0f;
    const std::size_t bytes = static_cast<std::size_t>(N) * q::q8_0_row_bytes(K);
    std::vector<std::uint8_t> store(bytes + offset + 1);
    std::uint8_t* wq = store.data() + offset;
    if (!q::quantize_q8_0_rows(w.data(), N, K, wq)) return 1e30;
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
            const std::uint8_t* blk = wq + n * q::q8_0_row_bytes(K) + 34 * b;
            const double sc = static_cast<double>(q::fp16_to_fp32(static_cast<std::uint16_t>(blk[0] | blk[1] << 8))) * dx[b];
            long long isum = 0, amag = 0;
            for (int i = 0; i < 32; ++i) {
                const long long p = static_cast<long long>(static_cast<std::int8_t>(blk[2 + i])) * xq[32 * b + i];
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

bool test_gemv(simd_ml::GemvQ8_0Fn fn, const char* path) {
    bool ok = true;
    auto run = [&](const char* what, int N, int K, int off, int fill) {
        const double w = gemv_case(fn, N, K, off, fill);
        const bool pass = w <= 1.0;
        printf("  %-12s %-7s N=%5d K=%5d off=%d  err/bound=%-8.3g %s\n", path, what, N, K, off, w, pass ? "PASS" : "FAIL");
        ok &= pass;
    };
    for (int N : {1, 2, 7, 8, 9, 15, 16, 17, 31, 100}) run("rows", N, 64, 0, 0);
    for (int K : {32, 96, 256, 1024, 4128}) run("k", 13, K, 0, 0);
    run("odd", 11, 160, 1, 0);
    run("extreme", 9, 4096, 3, 1);   // every quant ±127 and every |x| quant 127
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
    std::vector<std::uint8_t> wq(3 * q::q8_0_row_bytes(64));
    q::quantize_q8_0_rows(w.data(), 3, 64, wq.data());
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
    q::gemv_q8_0_run(N, K, x, Wq, y, Rows);
}

}  // namespace

int run_q8_0_tests() {
    printf("\n── Q8_0 Tests (quantize_q8_0_rows, KernelTable::gemv_q8_0) ──\n");
    printf("  active path: %s\n", q::q8_0_path());
    bool ok = test_fp16();
    ok &= test_quantize();
    ok &= for_each_kernel_table([](const simd_ml::KernelTable& t) { return test_gemv(t.gemv_q8_0, t.name); });
#if SIMD_ML_X86_AVAILABLE
    if (simd_ml::kernels_for(simd_ml::Isa::avx2)) {
        ok &= test_gemv(&via_driver<&q::q8_0_rows_avx2>, "avx2");
#  ifdef SIMD_ML_HAVE_AVXVNNI
        if (simd_ml::gemm::avx_vnni_available()) ok &= test_gemv(&via_driver<&q::q8_0_rows_avxvnni>, "avx_vnni");
        else printf("  avx_vnni: SKIP (CPU has no AVX-VNNI)\n");
#  endif
    }
#  ifdef SIMD_ML_HAVE_AVX512VNNI
    if (simd_ml::kernels_for(simd_ml::Isa::avx512) && simd_ml::gemm::avx512vnni_available())
        ok &= test_gemv(&via_driver<&q::q8_0_rows_avx512vnni>, "avx512_vnni");
    else
        printf("  avx512_vnni: SKIP (CPU has no AVX512_VNNI)\n");
#  endif
#endif
    printf("  Overall: %s\n", ok ? "ALL PASS" : "SOME FAILURES");
    return ok ? 0 : 1;
}
