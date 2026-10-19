/**
 * q4_0_scalar.cpp — Q4_0 quantization (ggml's quantize_row_q4_0_ref, bit for
 * bit) and the scalar table's Q4_0 GEMV. Part of the scalar unit: baseline
 * flags, no -ffast-math, and -ffp-contract=off (CMakeLists.txt), so
 * max / −8, 1/d, w·id and + 8.5 are each one IEEE binary32 operation, as in
 * ggml's C reference built for baseline x86-64 (which has no FMA to fuse
 * w·id + 8.5 into; a fused version truncates differently at the integers).
 */

#include "q4_0.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace simd_ml {
namespace quant {

namespace {

std::uint32_t float_bits(float f) noexcept { std::uint32_t u; std::memcpy(&u, &f, 4); return u; }

// q = min(15, trunc(v + 8.5)). v = w·id lies in [-8(1 + 2^-23), 8(1 + 2^-23)]
// (|w| <= |max|), so v + 8.5 is in (0, 17): the conversion is defined, and
// truncating toward zero equals ggml's (int8_t) cast.
inline std::uint8_t nibble(float v) noexcept {
    const float t = v + 8.5f;
    const int q = static_cast<int>(t);
    return static_cast<std::uint8_t>(q < 15 ? q : 15);
}

// One block into 16 bytes of nibbles; the FP32 scale d in *d_out. False if a value is NaN/±Inf.
inline bool quantize_block(const float* w, std::uint8_t* qs, float* d_out) noexcept {
    float amax = 0.0f, max = 0.0f;
    bool finite = true;
    for (int j = 0; j < kQ4_0Values; ++j) {
        // Bit test, not std::isfinite: the same check whatever the flags.
        finite &= (float_bits(w[j]) & 0x7f800000u) != 0x7f800000u;
        // Strictly larger: the first value of the largest magnitude keeps its
        // sign (ggml: `if (amax < fabsf(v))`), so d's sign follows it.
        if (amax < std::fabs(w[j])) {
            amax = std::fabs(w[j]);
            max = w[j];
        }
    }
    if (!finite) return false;
    const float d = max / -8.0f;   // a zero block gives d = -0.0 (stored 0x8000), as ggml
    float id = d != 0.0f ? 1.0f / d : 0.0f;
    // |max| below 8/FLT_MAX: 1/d overflows and ggml's C reference converts
    // ±inf or NaN to int8 (undefined). Use id = 0 (every q = 8, value 0); the
    // scale rounds to binary16 zero there anyway.
    if (std::fabs(id) > std::numeric_limits<float>::max()) id = 0.0f;
    for (int j = 0; j < kQ4_0Values / 2; ++j) {
        const float v0 = w[j] * id;
        const float v1 = w[kQ4_0Values / 2 + j] * id;
        qs[j] = static_cast<std::uint8_t>(nibble(v0) | (nibble(v1) << 4));
    }
    *d_out = d;
    return true;
}

}  // namespace

bool quantize_q4_0_rows(const float* W, int N, int K, std::uint8_t* out) noexcept {
    const int nb = K / kQ4_0Values;
    for (int n = 0; n < N; ++n) {
        const float* w = W + static_cast<std::size_t>(n) * K;
        for (int b = 0; b < nb; ++b, out += kQ4_0Bytes) {
            float d;
            if (!quantize_block(w + kQ4_0Values * b, out + 2, &d)) return false;
            const std::uint16_t h = fp32_to_fp16(d);
            if ((h & 0x7fffu) == 0x7c00u) return false;   // scale overflows binary16
            out[0] = static_cast<std::uint8_t>(h & 0xffu);   // little-endian, whatever the host
            out[1] = static_cast<std::uint8_t>(h >> 8);
        }
    }
    return true;
}

namespace {

void q4_0_rows_scalar(int N, int nb, const std::int8_t* xq, const float* dx,
                      const std::uint8_t* Wq, float* y) noexcept {
    const std::size_t row = static_cast<std::size_t>(nb) * kQ4_0Bytes;
    for (int n = 0; n < N; ++n) {
        const std::uint8_t* blk = Wq + n * row;
        float acc = 0.0f;
        for (int b = 0; b < nb; ++b, blk += kQ4_0Bytes) {
            const std::uint8_t* q = blk + 2;
            const std::int8_t* p = xq + kQ4_0Values * b;
            int isum = 0;   // exact: |isum| <= 32·8·127
            for (int j = 0; j < kQ4_0Values / 2; ++j)
                isum += ((q[j] & 0x0f) - 8) * p[j] + ((q[j] >> 4) - 8) * p[kQ4_0Values / 2 + j];
            const float dw = fp16_to_fp32(static_cast<std::uint16_t>(blk[0] | (blk[1] << 8)));
            acc += static_cast<float>(isum) * (dw * dx[b]);
        }
        y[n] = acc;
    }
}

}  // namespace

void gemv_q4_0_scalar(int N, int K, const float* x, const std::uint8_t* Wq, float* y) {
    gemv_q8_0_run(N, K, x, Wq, y, &q4_0_rows_scalar);
}

}  // namespace quant
}  // namespace simd_ml
