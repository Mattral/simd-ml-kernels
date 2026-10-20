/**
 * test_runner.cpp — C++ test runner (no framework). See tests/CMakeLists.txt.
 *
 * The suites run each kernel on every kernel table the CPU supports
 * (kernel_tables.hpp), so the same binary covers scalar-only builds too.
 */

#include <cstdio>

int run_alignment_tests();
int run_dispatch_tests();
int run_gelu_tests();
int run_gemm_packed_tests();
int run_lowp_gemm_tests();
int run_int8_gemm_tests();
int run_q8_0_tests();
int run_q4_0_tests();
int run_activation_tests();
int run_thread_tests();

int main() {
    std::printf("simd-ml-kernels C++ tests\n");
    int result = 0;
    result |= run_alignment_tests();
    result |= run_dispatch_tests();
    result |= run_gelu_tests();
    result |= run_gemm_packed_tests();
    result |= run_lowp_gemm_tests();
    result |= run_int8_gemm_tests();
    result |= run_q8_0_tests();
    result |= run_q4_0_tests();
    result |= run_activation_tests();
    result |= run_thread_tests();
    std::printf("\n%s\n", result == 0 ? "=== ALL TESTS PASSED ===" : "=== SOME TESTS FAILED ===");
    return result;
}
