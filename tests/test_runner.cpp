/**
 * test_runner.cpp — C++ test runner (no framework). See tests/CMakeLists.txt.
 *
 * The suites run each kernel on every kernel table the CPU supports
 * (kernel_tables.hpp). SIMD_ML_X86_AVAILABLE (set by CMake) is 0 when the
 * AVX2/AVX-512 units are not built; only the avx_matmul suite depends on it.
 */

#ifndef SIMD_ML_X86_AVAILABLE
#  define SIMD_ML_X86_AVAILABLE 0
#endif

#include <cstdio>

int run_alignment_tests();
int run_dispatch_tests();
int run_gelu_tests();
int run_gemm_packed_tests();
int run_f16_gemm_tests();
int run_bf16_gemm_tests();
int run_activation_tests();
int run_thread_tests();
#if SIMD_ML_X86_AVAILABLE
int run_gemm_tests();
#endif

int main() {
    std::printf("simd-ml-kernels C++ tests\n");
    int result = 0;
    result |= run_alignment_tests();
    result |= run_dispatch_tests();
    result |= run_gelu_tests();
#if SIMD_ML_X86_AVAILABLE
    result |= run_gemm_tests();
#endif
    result |= run_gemm_packed_tests();
    result |= run_f16_gemm_tests();
    result |= run_bf16_gemm_tests();
    result |= run_activation_tests();
    result |= run_thread_tests();
    std::printf("\n%s\n", result == 0 ? "=== ALL TESTS PASSED ===" : "=== SOME TESTS FAILED ===");
    return result;
}
