#pragma once

/**
 * naive_gemm.hpp — Scalar GEMM reference implementation.
 *
 * This file provides a simple correctness oracle for packed GEMM and
 * fallback paths. It uses row-major storage and a BLAS-compatible API.
 */

#include <cstddef>

namespace simd_ml {
namespace gemm_ref {

inline void naive_sgemm(int M, int N, int K,
                        float alpha,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float beta,
                        float* C, int ldc) {
    // K == 0 is valid (BLAS semantics): the sum is empty, so C = beta·C.
    if (M <= 0 || N <= 0 || K < 0) {
        return;
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            // beta == 0 overwrites C without reading it, so NaN/Inf or
            // uninitialised memory in C does not propagate (BLAS semantics).
            const float c_old = (beta == 0.0f) ? 0.0f : beta * C[i * ldc + j];
            C[i * ldc + j] = alpha * sum + c_old;
        }
    }
}

} // namespace gemm_ref
} // namespace simd_ml

inline void naive_sgemm(int M, int N, int K,
                        float alpha,
                        const float* A, int lda,
                        const float* B, int ldb,
                        float beta,
                        float* C, int ldc) noexcept {
    simd_ml::gemm_ref::naive_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
