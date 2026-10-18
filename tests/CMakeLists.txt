# tests/CMakeLists.txt — C++ test targets
#
# simd_tests links simd_ml_kernels_lib, the same objects that ship in the
# Python extension. The test code itself (float64 references, tolerance
# checks) is compiled with BASE_FLAGS only: no -mavx*, no -ffast-math. It
# reaches the SIMD kernels through the kernel tables (dispatch/kernels.hpp)
# and runs each test on every table this CPU supports, so the binary runs on
# any x86-64 CPU and, under Intel SDE, covers the AVX2 and AVX-512 paths too.
#
# Sanitizers: configure with -DSIMD_ML_SANITIZE=ON. The top-level CMakeLists
# then instruments every target, kernels included.

set(TEST_SRCS
    test_runner.cpp
    test_alignment.cpp
    test_dispatch.cpp
    test_gelu.cpp
    test_gemm_packed.cpp
    test_gemm_lowp.cpp
    test_gemm_int8.cpp
    test_activations.cpp
    test_threads.cpp
)
add_executable(simd_tests ${TEST_SRCS})
target_include_directories(simd_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels
    ${CMAKE_SOURCE_DIR}/src
)
if(SIMD_ML_ENABLE_X86)
    target_compile_definitions(simd_tests PRIVATE SIMD_ML_X86_AVAILABLE=1)
    # Extension units in this build (CMakeLists.txt): their entry points are
    # called directly by the tests.
    foreach(ext IN LISTS _ext_units)
        if(ext IN_LIST SIMD_ML_ISAS)
            string(TOUPPER ${ext} EXT)
            target_compile_definitions(simd_tests PRIVATE SIMD_ML_HAVE_${EXT})
        endif()
    endforeach()
else()
    target_compile_definitions(simd_tests PRIVATE SIMD_ML_X86_AVAILABLE=0)
endif()
target_link_libraries(simd_tests PRIVATE simd_ml_kernels_lib)
target_compile_options(simd_tests PRIVATE ${BASE_FLAGS})
add_test(NAME simd_tests COMMAND simd_tests)

# ─── ISA units contain only their own ISA ─────────────────────────────────────
# check_isa_units.py disassembles the object files: the scalar unit (dispatch
# and scalar kernels) must contain no VEX/EVEX instruction, the AVX2 units no
# EVEX (AVX-512) instruction, and no symbol may be emitted out of line by two
# units built with different flags (an inline or template function the linker
# could pick from the wrong unit). The bindings are compiled with BASE_FLAGS
# too; the SDE runs in CI (-nhm: no AVX) check them at run time.
if(SIMD_ML_ENABLE_X86 AND CMAKE_OBJDUMP AND CMAKE_NM
   AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT SIMD_ML_SANITIZE)
    set(_units "")
    foreach(isa IN LISTS SIMD_ML_ISAS)
        list(APPEND _units "--unit=${isa}=$<JOIN:$<TARGET_OBJECTS:simd_ml_isa_${isa}>,|>")
    endforeach()
    add_test(NAME isa_units
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_isa_units.py
                     --objdump ${CMAKE_OBJDUMP} --nm ${CMAKE_NM} ${_units})
endif()
