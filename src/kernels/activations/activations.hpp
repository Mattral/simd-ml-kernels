#pragma once

/**
 * activations.hpp — Public API for vectorized activation functions.
 *
 * The AVX2 kernels (AVX2 units; call them through dispatch/kernels.hpp,
 * which checks the CPU). The scalar versions are in scalar_kernels.cpp.
 */

#include <cstddef>

namespace activations {

/**
 * relu_avx2 — AVX2-vectorized ReLU: output[i] = max(0, input[i]).
 */
void relu_avx2(const float* input, float* output, int n);

/**
 * silu_avx2 — AVX2-vectorized SiLU/Swish: output[i] = x / (1 + exp(-x)).
 * |err| <= 1e-35 + 1e-6*|ref| for every finite x (see silu_avx2.cpp).
 */
void silu_avx2(const float* input, float* output, int n);

/**
 * softmax_row_avx2 — Numerically stable softmax over a single row of n floats.
 */
void softmax_row_avx2(const float* input, float* output, int n);

/**
 * layer_norm_avx2 — Layer normalization over a row of n elements.
 * output[i] = (input[i] - mean) / sqrt(variance + eps) * gamma[i] + beta[i]
 * gamma and beta may be nullptr for the raw normalized output.
 */
void layer_norm_avx2(const float* input, float* output, int n,
                     const float* gamma, const float* beta,
                     float eps = 1e-5f);

/**
 * rms_norm_avx2 — RMSNorm of one row (Llama, T5):
 *   output[i] = weight[i] * (input[i] * r),  r = 1 / sqrt(Σx²/n + eps)
 * in that order of operations (as transformers' LlamaRMSNorm); weight may be
 * nullptr. Σx² accumulates in float lanes added into a double every 1024
 * elements. |err| <= 1e-30 + 1e-6*|ref| against float64 (rms_norm_avx2.cpp).
 */
void rms_norm_avx2(const float* input, float* output, int n, const float* weight, float eps);

/**
 * silu_mul_avx2 — output[i] = SiLU(gate[i]) * up[i], the gated MLP product of
 * Llama's SwiGLU, in one pass (SiLU as silu_avx2, then one multiply).
 * |err| <= 1e-35*|up| + 1.1e-6*|ref| for finite inputs (silu_avx2.cpp).
 */
void silu_mul_avx2(const float* gate, const float* up, float* output, int n);

} // namespace activations
