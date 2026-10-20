/**
 * q8_0_scalar.cpp — Q8_0 quantization (ggml's quantize_row_q8_0_ref, bit for
 * bit), the GEMV driver shared by every table, and the scalar table's GEMV.
 * Part of the scalar unit: baseline flags, no -ffast-math, so the divisions,
 * the multiply by 1/d and the rounding are IEEE operations exactly as in
 * ggml's C reference (a fast-math unit could turn amax / 127 into
 * amax · (1/127), which rounds differently). Quantizing x costs K values per
 * call against N·K weights read, under 0.1% of a decode GEMV's bytes.
 */

#include "q8_0.hpp"
#include "../../dispatch/kernels.hpp"
#include "../gemm/avx2_gemm_packed.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#ifdef SIMD_ML_OPENMP
#  include <omp.h>
#endif

namespace simd_ml {
namespace quant {

namespace {

std::uint32_t float_bits(float f) noexcept { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }
float bits_float(std::uint32_t u) noexcept { float f; std::memcpy(&f, &u, 4); return f; }

// roundf (half away from zero) for |v| <= 127.5: v - trunc(v) is exact, and
// the conversions are exact, so this equals libm roundf without a call per
// value (baseline x86-64 has no roundss) and auto-vectorizes.
inline std::int8_t round_q(float v) noexcept {
    const float t = static_cast<float>(static_cast<int>(v));
    const float r = v - t;
    return static_cast<std::int8_t>(static_cast<int>(t) + (r >= 0.5f) - (r <= -0.5f));
}

// One block: amax, d = amax / 127, q = round(w · id). False if a value is NaN/±Inf.
inline bool quantize_block(const float* w, std::int8_t* q, float* d_out) noexcept {
    float amax = 0.0f;
    bool finite = true;
    for (int j = 0; j < kQ8_0Values; ++j) {
        // Bit test, not std::isfinite: the same check whatever the flags.
        finite &= (float_bits(w[j]) & 0x7f800000u) != 0x7f800000u;
        amax = std::max(amax, std::fabs(w[j]));
    }
    if (!finite) return false;
    const float d = amax / 127.0f;
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    // amax below 127/FLT_MAX: 1/d overflows and ggml's C reference converts
    // ±inf or NaN (0·inf) to int8, undefined in C (0 on x86). Store q = 0;
    // the scale rounds to binary16 zero there anyway.
    if (id > std::numeric_limits<float>::max()) id = 0.0f;
    for (int j = 0; j < kQ8_0Values; ++j) q[j] = round_q(w[j] * id);
    *d_out = d;
    return true;
}

}  // namespace

std::uint16_t fp32_to_fp16(float f) noexcept {
    std::uint32_t w = float_bits(f);
    const std::uint16_t sign = static_cast<std::uint16_t>((w >> 16) & 0x8000u);
    w &= 0x7fffffffu;
    if (w >= 0x7f800000u) return sign | (w > 0x7f800000u ? 0x7e00u : 0x7c00u);  // NaN, ±Inf
    if (w >= 0x477ff000u) return sign | 0x7c00u;   // >= 65520: rounds (to even) to infinity
    if (w < 0x38800000u) {
        // Below 2^-14: a binary16 subnormal, k·2^-24. Adding 0.5 (whose ulp is
        // 2^-24) rounds |f| to the nearest multiple of 2^-24, ties to even;
        // the low bits are k (k = 1024 gives 0x0400, the smallest normal).
        return sign | static_cast<std::uint16_t>(float_bits(bits_float(w) + 0.5f) - 0x3f000000u);
    }
    // Normal: rebias the exponent (127 -> 15) and round the 13 dropped bits to
    // nearest even; a mantissa carry moves into the exponent, as it should.
    w += 0xc8000fffu + ((w >> 13) & 1u);
    return sign | static_cast<std::uint16_t>(w >> 13);
}

float fp16_to_fp32(std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    if (exp == 0x1f) return bits_float(sign | 0x7f800000u | (man << 13));
    if (exp == 0) {
        if (man == 0) return bits_float(sign);
        exp = 113;   // subnormal man·2^-24: normalize
        while ((man & 0x400u) == 0) { man <<= 1; --exp; }
        return bits_float(sign | (exp << 23) | ((man & 0x3ffu) << 13));
    }
    return bits_float(sign | ((exp + 112) << 23) | (man << 13));
}

bool quantize_q8_0_rows(const float* W, int N, int K, std::uint8_t* out) noexcept {
    const int nb = K / kQ8_0Values;
    for (int n = 0; n < N; ++n) {
        const float* w = W + static_cast<std::size_t>(n) * K;
        for (int b = 0; b < nb; ++b, out += kQ8_0Bytes) {
            float d;
            if (!quantize_block(w + kQ8_0Values * b, reinterpret_cast<std::int8_t*>(out + 2), &d))
                return false;
            const std::uint16_t h = fp32_to_fp16(d);
            if ((h & 0x7fffu) == 0x7c00u) return false;   // scale overflows binary16
            out[0] = static_cast<std::uint8_t>(h & 0xffu);   // little-endian, whatever the host
            out[1] = static_cast<std::uint8_t>(h >> 8);
        }
    }
    return true;
}

bool quantize_x_q8(const float* x, int K, std::int8_t* qs, float* dx) noexcept {
    const int nb = K / kQ8_0Values;
    for (int b = 0; b < nb; ++b)
        if (!quantize_block(x + kQ8_0Values * b, qs + kQ8_0Values * b, dx + b)) return false;
    return true;
}

void gemv_q8_0_run(int N, int K, const float* x, const std::uint8_t* Wq, float* y, Q8RowsFn rows,
                   int block_bytes) {
    if (N <= 0) return;
    if (K <= 0) { std::fill(y, y + N, 0.0f); return; }
    const int nb = K / kQ8_0Values;
    // Per-thread scratch, kept between calls (x is K values: at most a few
    // tens of KiB for LLM widths).
    thread_local std::vector<std::int8_t> qs;
    thread_local std::vector<float> dx;
    if (qs.size() < static_cast<std::size_t>(K)) qs.resize(static_cast<std::size_t>(K));
    if (dx.size() < static_cast<std::size_t>(nb)) dx.resize(static_cast<std::size_t>(nb));
    const std::size_t row = static_cast<std::size_t>(nb) * block_bytes;
#ifdef SIMD_ML_OPENMP
    // x is quantized once per call, by the team: each thread quantizes one
    // contiguous run of 32-value blocks with the same scalar code (blocks are
    // independent, so the quants are the ones one thread would write), then
    // a barrier, then each thread reads all of x for its rows. On one thread
    // the quantization ran at ~1.6 ns per value (13.5 µs at K = 8192), which
    // was most of a 4-thread Q4_0 512×2048 call (DESIGN.md §2.10).
    if (const int threads = gemm::gemv_threads(N, static_cast<double>(N) * row); threads > 1) {
        std::int8_t* xq = qs.data();
        float* xd = dx.data();
        std::atomic<bool> bad{false};   // x holds a NaN or ±Inf: every y is NaN
#pragma omp parallel num_threads(threads)
        {
            const int t = omp_get_thread_num(), team = omp_get_num_threads();
            const int b0 = static_cast<int>(static_cast<long long>(nb) * t / team);
            const int b1 = static_cast<int>(static_cast<long long>(nb) * (t + 1) / team);
            if (!quantize_x_q8(x + kQ8_0Values * b0, kQ8_0Values * (b1 - b0), xq + kQ8_0Values * b0, xd + b0))
                bad.store(true, std::memory_order_relaxed);
#pragma omp barrier
            const gemm::GemvSlice s = gemm::gemv_slice(N, t, team);
            if (bad.load(std::memory_order_relaxed))
                std::fill(y + s.lo, y + s.hi, std::numeric_limits<float>::quiet_NaN());
            else if (s.hi > s.lo)
                rows(s.hi - s.lo, nb, xq, xd, Wq + s.lo * row, y + s.lo);
        }
        return;
    }
#endif
    (void)row;
    if (!quantize_x_q8(x, K, qs.data(), dx.data())) {
        std::fill(y, y + N, std::numeric_limits<float>::quiet_NaN());
        return;
    }
    rows(N, nb, qs.data(), dx.data(), Wq, y);
}

namespace {

void q8_0_rows_scalar(int N, int nb, const std::int8_t* xq, const float* dx,
                      const std::uint8_t* Wq, float* y) noexcept {
    const std::size_t row = static_cast<std::size_t>(nb) * kQ8_0Bytes;
    for (int n = 0; n < N; ++n) {
        const std::uint8_t* blk = Wq + n * row;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b, blk += kQ8_0Bytes) {
            const auto* q = reinterpret_cast<const std::int8_t*>(blk + 2);
            const std::int8_t* p = xq + kQ8_0Values * b;
            int isum = 0;   // exact: |isum| <= 32·127·127
            for (int i = 0; i < kQ8_0Values; ++i) isum += q[i] * p[i];
            const float dw = fp16_to_fp32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
            acc += static_cast<float>(isum) * (dw * dx[b]);
        }
        y[n] = acc;
    }
}

}  // namespace

void gemv_q8_0_scalar(int N, int K, const float* x, const std::uint8_t* Wq, float* y) {
    gemv_q8_0_run(N, K, x, Wq, y, &q8_0_rows_scalar, kQ8_0Bytes);
}

const char* q8_0_path() noexcept {
    const KernelTable& t = kernels();
    const dispatch::CpuFeatures& f = cpu_features();
    if (t.isa == Isa::scalar) return "scalar";
#ifdef SIMD_ML_HAVE_AVX512VNNI
    if (t.isa == Isa::avx512 && f.has_avx512vnni) return "avx512_vnni";
#endif
#ifdef SIMD_ML_HAVE_AVXVNNI
    if (f.has_avx_vnni) return "avx_vnni";
#endif
    (void)f;
    return "avx2";
}

}  // namespace quant
}  // namespace simd_ml
