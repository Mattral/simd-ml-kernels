# tests/CMakeLists.txt — C++ test targets
#
# simd_tests links exactly one kernel library: simd_kernels_lib, compiled with
# the same flags that ship in the Python extension, so the tests check the code
# users run. The test sources themselves (float64 references, tolerance checks)
# are compiled with PRECISION_FLAGS, i.e. without -ffast-math.
#
# There used to be a second library, simd_kernels_precision_lib, that compiled
# the same kernel sources without -ffast-math and was linked after
# simd_kernels_lib. Two definitions of each kernel symbol is an ODR violation,
# and because the linker resolves each symbol from the first archive that
# provides it, every kernel object actually came from simd_kernels_lib
# (verified by comparing .text sizes in the map file); the precision library
# was never used.
#
# Sanitizers: configure with -DSIMD_ML_SANITIZE=ON. The top-level CMakeLists
# then instruments every target, kernels included, so simd_tests itself is the
# ASan/UBSan run.

# ─── doctest dependency ───────────────────────────────────────────────────────
find_package(doctest CONFIG QUIET)
if(NOT doctest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG v2.4.12
        # EXCLUDE_FROM_ALL: prevents doctest's own install() rules from running
        # when the user runs `cmake --install`. Only our project artifacts install.
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(doctest)
endif()

# ─── Main C++ test runner ─────────────────────────────────────────────────────
# On non-x86 hosts only the alignment test and a stub runner are compiled.
# All AVX2-specific test files (gelu, gemm, gemm_packed, activations, threads)
# include x86 intrinsic headers directly and must be excluded on ARM.
# test_runner.cpp uses SIMD_ML_X86_AVAILABLE to guard their call-sites.
set(TEST_SRCS_COMMON
    test_runner.cpp
    test_alignment.cpp
)
set(TEST_SRCS_X86
    test_gelu.cpp
    test_gemm.cpp
    test_gemm_packed.cpp
    test_gemm_f16.cpp
    test_gemm_bf16.cpp
    test_activations.cpp
    test_threads.cpp
)
if(SIMD_ML_ENABLE_X86)
    set(TEST_SRCS ${TEST_SRCS_COMMON} ${TEST_SRCS_X86})
else()
    set(TEST_SRCS ${TEST_SRCS_COMMON})
endif()
add_executable(simd_tests ${TEST_SRCS})
target_include_directories(simd_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels
    ${CMAKE_SOURCE_DIR}/src
)
# Pass x86 availability as a preprocessor constant so test_runner.cpp can
# conditionally call run_gelu_tests(), run_gemm_tests(), etc.
if(SIMD_ML_ENABLE_X86)
    target_compile_definitions(simd_tests PRIVATE SIMD_ML_X86_AVAILABLE=1)
    # test_gemm_f16.cpp uses F16C intrinsics (helper converters); give it
    # the same -mf16c flag we pass to f16_gemm.cpp in the kernel library.
    if(COMPILER_HAS_F16C)
        set_source_files_properties(test_gemm_f16.cpp
            PROPERTIES COMPILE_OPTIONS "-mf16c")
    endif()
else()
    target_compile_definitions(simd_tests PRIVATE SIMD_ML_X86_AVAILABLE=0)
endif()
target_link_libraries(simd_tests PRIVATE simd_kernels_lib)
target_compile_options(simd_tests PRIVATE ${PRECISION_FLAGS})
add_test(NAME simd_tests COMMAND simd_tests)
