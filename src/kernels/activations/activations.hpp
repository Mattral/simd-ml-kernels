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
 * silu_avx2 — AVX2-vectorized SiLU/Swish: output[i] = x * sigmoid(x).
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

} // namespace activations
