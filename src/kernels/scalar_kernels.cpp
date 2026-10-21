/**
 * scalar_kernels.cpp — portable C++ versions of every kernel, and the scalar
 * kernel table.
 *
 * Used on CPUs without AVX2 + FMA + F16C (or with SIMD_ML_ISA=scalar), and on
 * non-x86 targets. Compiled with the baseline flags only and without
 * -ffast-math, so libm handles NaN/Inf with IEEE semantics. They are written
 * for clarity, not speed: the compiler may auto-vectorise the inner loops for
 * the baseline ISA (SSE2 on x86-64), nothing more.
 *
 * Results match the AVX2 kernels within the tolerances the tests assert, and
 * have the same special-value behaviour where the tests pin it down (GeLU of
 * ±inf and x < -10, softmax rows with NaN/±inf, beta = 0 never reading C).
 */

#include "../dispatch/kernels.hpp"
#include "activations/activations.hpp"
#include "intrinsic_gelu.hpp"
#include "quant/q4_0.hpp"
#include "quant/q8_0.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#ifndef SIMD_ML_UNIT_FLAGS
#  define SIMD_ML_UNIT_FLAGS "unknown"
#endif

namespace {

// C = beta*C, with beta = 0 writing zeros so NaN/garbage in C does not propagate.
void scale_c(float* C, int M, int N, int ldc, float beta) {
    if (beta == 1.0f) return;
    for (int i = 0; i < M; ++i) {
        float* row = C + static_cast<std::size_t>(i) * ldc;
        if (beta == 0.0f) std::fill(row, row + N, 0.0f);
        else for (int j = 0; j < N; ++j) row[j] *= beta;
    }
}

// C += alpha * A·B with A and B converted to float by `load`. i-p-j order: the
// inner loop runs over contiguous rows of B and C (unit stride).
template <typename T, typename Load>
void gemm_rows(int M, int N, int K, float alpha, const T* A, int lda, const T* B, int ldb,
               float beta, float* C, int ldc, Load load) {
    if (M <= 0 || N <= 0 || K < 0) return;
    scale_c(C, M, N, ldc, beta);
    if (K == 0) return;
    std::vector<float> acc(static_cast<std::size_t>(N));
    for (int i = 0; i < M; ++i) {
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int p = 0; p < K; ++p) {
            const float a = load(A[static_cast<std::size_t>(i) * lda + p]);
            const T* b = B + static_cast<std::size_t>(p) * ldb;
            for (int j = 0; j < N; ++j) acc[j] += a * load(b[j]);
        }
        float* c = C + static_cast<std::size_t>(i) * ldc;
        for (int j = 0; j < N; ++j) c[j] += alpha * acc[j];
    }
}

// C += alpha * A·Bᵀ with B stored N×K (x·Wᵀ): one dot product along contiguous
// rows of A and B per element of C, as the SIMD tables' GEMV does.
template <typename T, typename Load>
void gemm_rows_tb(int M, int N, int K, float alpha, const T* A, int lda, const T* B, int ldb,
                  float beta, float* C, int ldc, Load load) {
    if (M <= 0 || N <= 0 || K < 0) return;
    scale_c(C, M, N, ldc, beta);
    if (K == 0) return;
    for (int i = 0; i < M; ++i) {
        const T* a = A + static_cast<std::size_t>(i) * lda;
        float* c = C + static_cast<std::size_t>(i) * ldc;
        for (int j = 0; j < N; ++j) {
            const T* b = B + static_cast<std::size_t>(j) * ldb;
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) acc += load(a[p]) * load(b[p]);
            c[j] += alpha * acc;
        }
    }
}

float bits_to_float(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

// IEEE binary16 -> binary32, exact for every input (subnormals, ±inf, NaN).
float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t man = h & 0x3ffu;
    if (exp == 0x1f) return bits_to_float(sign | 0x7f800000u | (man << 13));
    if (exp == 0) {
        if (man == 0) return bits_to_float(sign);
        // Subnormal man * 2^-24: shift until the implicit bit (bit 10) is set;
        // the float exponent starts at 127 - 14 = 113 and drops by one per shift.
        exp = 113;
        while ((man & 0x400u) == 0) { man <<= 1; --exp; }
        return bits_to_float(sign | (exp << 23) | ((man & 0x3ffu) << 13));
    }
    return bits_to_float(sign | ((exp + 112) << 23) | (man << 13));
}

void sgemm_scalar(bool trans_a, bool trans_b, int M, int N, int K, float alpha,
                  const float* A, int lda, const float* B, int ldb, float beta,
                  float* C, int ldc) {
    if (!trans_a && !trans_b) {
        gemm_rows(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, [](float v) { return v; });
        return;
    }
    if (!trans_a) {
        gemm_rows_tb(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, [](float v) { return v; });
        return;
    }
    // op(A)(i, p), op(B)(p, j) with BLAS transposes; same summation order as gemm_rows.
    const auto a_at = [&](int i, int p) {
        return trans_a ? A[static_cast<std::size_t>(p) * lda + i] : A[static_cast<std::size_t>(i) * lda + p];
    };
    const auto b_at = [&](int p, int j) {
        return trans_b ? B[static_cast<std::size_t>(j) * ldb + p] : B[static_cast<std::size_t>(p) * ldb + j];
    };
    std::vector<float> acc(static_cast<std::size_t>(N));
    for (int i = 0; i < M; ++i) {
        float* c = C + static_cast<std::size_t>(i) * ldc;
        for (int j = 0; j < N; ++j) c[j] = beta == 0.0f ? 0.0f : beta * c[j];
        std::fill(acc.begin(), acc.end(), 0.0f);
        for (int p = 0; p < K; ++p) {
            const float a = a_at(i, p);
            for (int j = 0; j < N; ++j) acc[j] += a * b_at(p, j);
        }
        for (int j = 0; j < N; ++j) c[j] += alpha * acc[j];
    }
}

void sgemm_f16_scalar(bool trans_b, int M, int N, int K, float alpha, const std::uint16_t* A, int lda,
                      const std::uint16_t* B, int ldb, float beta, float* C, int ldc) {
    if (trans_b)
        gemm_rows_tb(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, half_to_float);
    else
        gemm_rows(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, half_to_float);
}

// INT8 GEMM (gemm/int8_gemm.hpp): exact int64 sums, one rounding at the end.
void sgemm_u8s8_scalar(bool trans_b, int M, int N, int K, const std::uint8_t* A, int lda,
                       int a_zero_point, const std::int8_t* B, int ldb, float a_scale,
                       const float* b_scales, float* C, int ldc) {
    if (M <= 0 || N <= 0 || K < 0) return;
    std::vector<long long> acc(static_cast<std::size_t>(N));
    for (int i = 0; i < M; ++i) {
        std::fill(acc.begin(), acc.end(), 0LL);
        const std::uint8_t* a = A + static_cast<std::size_t>(i) * lda;
        for (int p = 0; p < K; ++p) {
            const long long av = static_cast<long long>(a[p]) - a_zero_point;
            if (trans_b) {
                for (int j = 0; j < N; ++j) acc[j] += av * B[static_cast<std::size_t>(j) * ldb + p];
            } else {
                const std::int8_t* b = B + static_cast<std::size_t>(p) * ldb;
                for (int j = 0; j < N; ++j) acc[j] += av * b[j];
            }
        }
        float* c = C + static_cast<std::size_t>(i) * ldc;
        for (int j = 0; j < N; ++j)
            c[j] = static_cast<float>(static_cast<double>(a_scale) * b_scales[j] * static_cast<double>(acc[j]));
    }
}

void sgemm_bf16_scalar(bool trans_b, int M, int N, int K, float alpha, const std::uint16_t* A, int lda,
                       const std::uint16_t* B, int ldb, float beta, float* C, int ldc) {
    const auto load = [](std::uint16_t v) { return bits_to_float(static_cast<std::uint32_t>(v) << 16); };
    if (trans_b)
        gemm_rows_tb(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, load);
    else
        gemm_rows(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, load);
}

void gelu_tanh_scalar(const float* in, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const float x = in[i];
        // Same cut-off as the AVX2 kernel: GeLU(x < -10) is -0 (|GeLU(-10)| =
        // 7.6e-23); without it, x = -inf gives -inf * 0 = NaN.
        if (x < -10.0f) { out[i] = -0.0f; continue; }
        const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void relu_scalar(const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

void silu_scalar(const float* in, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = in[i] / (1.0f + std::exp(-in[i]));
}

void softmax_row_scalar(const float* in, float* out, int n) {
    if (n <= 0) return;
    // As torch.softmax: a NaN or +inf anywhere, or every entry -inf, gives a
    // NaN row (exp(x - max) is NaN for all x then).
    float maxv = -std::numeric_limits<float>::infinity();
    bool nonfinite = false;
    for (int i = 0; i < n; ++i) {
        nonfinite |= std::isnan(in[i]) || in[i] == std::numeric_limits<float>::infinity();
        maxv = std::max(maxv, in[i]);
    }
    if (nonfinite || maxv == -std::numeric_limits<float>::infinity()) {
        std::fill(out, out + n, std::numeric_limits<float>::quiet_NaN());
        return;
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double e = std::exp(static_cast<double>(in[i]) - maxv);
        out[i] = static_cast<float>(e);
        sum += e;
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (int i = 0; i < n; ++i) out[i] *= inv;
}

void layer_norm_scalar(const float* in, float* out, int n,
                       const float* gamma, const float* beta, float eps) {
    if (n <= 0) return;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += in[i];
    const float mean = static_cast<float>(sum / n);
    double var = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = static_cast<double>(in[i]) - mean;
        var += d * d;
    }
    const float inv_std = 1.0f / std::sqrt(static_cast<float>(var / n) + eps);
    for (int i = 0; i < n; ++i) {
        float y = (in[i] - mean) * inv_std;
        if (gamma) y *= gamma[i];
        if (beta) y += beta[i];
        out[i] = y;
    }
}

void rms_norm_scalar(const float* in, float* out, int n, const float* weight, float eps) {
    if (n <= 0) return;
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += static_cast<double>(in[i]) * in[i];
    // In double, as rms_norm_avx2.cpp: no rsqrt estimate under -ffast-math.
    const float r = static_cast<float>(1.0 / std::sqrt(static_cast<double>(static_cast<float>(sum / n) + eps)));
    for (int i = 0; i < n; ++i) out[i] = weight ? weight[i] * (in[i] * r) : in[i] * r;
}

void silu_mul_scalar(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
}

}  // namespace

// Exact GeLU with libm erff: the scalar kernel and the tests' float32 oracle.
void gelu_forward_scalar(const float* input, float* output, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        const float x = input[i];
        if (x < -10.0f) { output[i] = -0.0f; continue; }   // see gelu_tanh_scalar
        output[i] = 0.5f * x * (1.0f + std::erf(x * 0.7071067811865476f));
    }
}

namespace simd_ml {

static void sgemm_blocking_scalar(int* kc, int* mc, int* nc) { *kc = *mc = *nc = 0; }

const KernelTable& kernel_table_scalar() noexcept {
    static const KernelTable t = {
        Isa::scalar, "scalar", SIMD_ML_UNIT_FLAGS, "row x 1 (no blocking)", &sgemm_blocking_scalar,
        &sgemm_scalar, &sgemm_f16_scalar, &sgemm_bf16_scalar, &sgemm_u8s8_scalar, &quant::gemv_q8_0_scalar,
        &quant::gemv_q4_0_scalar,
        &gelu_forward_scalar, &gelu_tanh_scalar,
        &relu_scalar, &silu_scalar, &softmax_row_scalar, &layer_norm_scalar,
        &rms_norm_scalar, &silu_mul_scalar,
    };
    return t;
}

}  // namespace simd_ml
