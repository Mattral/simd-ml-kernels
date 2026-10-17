/**
 * test_runner.cpp — Standalone C++ test runner (no framework dependencies).
 *
 * Links against simd_ml_kernels_lib only, built with the flags that ship.
 * See tests/CMakeLists.txt.
 *
 * SIMD_ML_X86_AVAILABLE is injected by CMake (1 on x86_64, 0 on ARM/other).
 * When it is 0 the x86/AVX2 test files are not compiled and their run_*()
 * functions do not exist, so we must not declare or call them.
 */

#ifndef SIMD_ML_X86_AVAILABLE
#  define SIMD_ML_X86_AVAILABLE 0
#endif

#include <cstdio>

// Forward declarations — always available
int run_alignment_tests();

// Forward declarations — x86/AVX2 only (not compiled on ARM etc.)
#if SIMD_ML_X86_AVAILABLE
int run_gelu_tests();
int run_gemm_tests();
int run_gemm_packed_tests();
int run_f16_gemm_tests();
int run_bf16_gemm_tests();
int run_activation_tests();
int run_thread_tests();
#endif

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║  simd-ml-kernels  ·  C++ Test Suite      ║\n");
#if SIMD_ML_X86_AVAILABLE
    printf("║  Architecture: x86_64 (AVX2/FMA path)   ║\n");
#else
    printf("║  Architecture: scalar fallback (non-x86) ║\n");
#endif
    printf("╚══════════════════════════════════════════╝\n");

    int result = 0;
    result |= run_alignment_tests();

#if SIMD_ML_X86_AVAILABLE
    result |= run_gelu_tests();
    result |= run_gemm_tests();
    result |= run_gemm_packed_tests();
    result |= run_f16_gemm_tests();
    result |= run_bf16_gemm_tests();
    result |= run_activation_tests();
    result |= run_thread_tests();
#else
    printf("[SKIP] x86/AVX2 test suites omitted on non-x86 host\n");
#endif

    printf("\n%s\n", result == 0
        ? "═══ ALL TESTS PASSED ═══"
        : "═══ SOME TESTS FAILED ═══");
    return result;
}
