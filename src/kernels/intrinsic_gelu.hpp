#pragma once

#include <cstddef>

/**
 * gelu_forward_avx2 — GeLU, tanh approximation (PyTorch approximate='tanh').
 * `input` and `output` may alias for in-place operation.
 */
void gelu_forward_avx2(const float* input,
                       float*       output,
                       std::size_t  n);

/**
 * gelu_erf_forward_avx2 — exact GeLU, 0.5·x·(1 + erf(x/√2)) (PyTorch
 * approximate='none'). `input` and `output` may alias.
 */
void gelu_erf_forward_avx2(const float* input,
                           float*       output,
                           std::size_t  n);

/**
 * gelu_forward_scalar — exact GeLU with libm erff (scalar_kernels.cpp): the
 * scalar kernel and the tests' float32 reference. x < -10 returns -0.
 */
void gelu_forward_scalar(const float* input,
                          float*       output,
                          std::size_t  n);
