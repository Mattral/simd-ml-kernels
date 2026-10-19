cmake_minimum_required(VERSION 3.22)
project(simd_ml_kernels VERSION 0.3.0 LANGUAGES CXX)
# CMake's VERSION is numeric only; the PEP 440 pre-release suffix of
# pyproject.toml's version goes here ("" for a release). build_info()["version"]
# is "${PROJECT_VERSION}${SIMD_ML_PRERELEASE}" and must equal the package version.
set(SIMD_ML_PRERELEASE "")

# Must be included before any use of CMAKE_INSTALL_INCLUDEDIR / LIBDIR / etc.
# (used in target_include_directories() below and in the install() rules further down).
include(GNUInstallDirs)

# ─── Compiler requirements ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Find dependencies ────────────────────────────────────────────────────────
# Development.Module, not Development: building an extension needs the
# headers, not libpython, which manylinux images do not ship.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11)
find_package(OpenMP OPTIONAL_COMPONENTS CXX)
option(SIMD_ML_OPENMP "Enable OpenMP multithreading in GEMM" OFF)
option(BENCH_OPENBLAS "Link against OpenBLAS for baseline comparison" OFF)
option(SIMD_ML_SANITIZE "Instrument every target (kernels included) with ASan + UBSan" OFF)
# Wheel builds (pyproject.toml) turn these off: a wheel only needs the Python
# extension.
option(SIMD_ML_BUILD_TESTS      "Build the C++ test suite"                ON)
option(SIMD_ML_BUILD_BENCHMARKS "Build bench and bench_stat (x86 only)"   ON)

# ─── Target ──────────────────────────────────────────────────────────────────
# x86-64 is detected from the compiler's own macros, not CMAKE_SYSTEM_PROCESSOR:
# that variable comes from the environment (PROCESSOR_ARCHITECTURE on Windows)
# and was empty in an MSYS2 shell, which silently produced a scalar-only build.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#if !(defined(__x86_64__) || defined(_M_X64))
#error not x86-64
#endif
int main() { return 0; }" SIMD_ML_TARGET_X86_64)
option(SIMD_ML_ENABLE_X86
    "Build the AVX2 and AVX-512 units (default: on when the compiler targets x86-64)"
    ${SIMD_ML_TARGET_X86_64})
if(SIMD_ML_ENABLE_X86 AND NOT SIMD_ML_TARGET_X86_64)
    message(WARNING "SIMD_ML_ENABLE_X86=ON but the compiler does not target x86-64; forcing OFF")
    set(SIMD_ML_ENABLE_X86 OFF CACHE BOOL "" FORCE)
endif()
option(SIMD_ML_NATIVE
    "Also compile every unit with -march=native (local benchmarking only: the result runs only on CPUs like the build host)"
    OFF)

# ─── Compiler flags ───────────────────────────────────────────────────────────
# Every unit gets BASE_FLAGS. The ISA units (below) add their ISA's flags and
# nothing else, so no instruction outside that ISA can appear in them, and code
# outside them (dispatch, scalar kernels, bindings, tests) runs on any x86-64
# CPU (SSE2). Which unit runs is decided at run time (src/dispatch/kernels.hpp).
#
# The AVX2 and AVX-512 kernel units get -ffast-math -fno-finite-math-only:
# reassociation and the other fast-math rewrites, but NaN and Inf keep IEEE
# semantics, so isnan/isinf tests and -inf masks cannot be optimised away.
# Measured on the reference laptop (per-call minimum, 64 Ki elements, three
# alternating rounds; PR #18): as fast as or faster than plain -ffast-math on
# every kernel (exact GeLU 20 vs 23 us, ReLU 4.4 vs 6.9 us) and faster than no
# fast-math (LayerNorm 14 vs 18 us); FP32 GEMM is unaffected. The scalar
# kernels, dispatch code, bindings and tests are compiled without fast-math.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set(WARN_FLAGS -Wall -Wextra -Wshadow -Wno-unused-parameter)
    set(BASE_FLAGS -O3 -DNDEBUG ${WARN_FLAGS})
    if(SIMD_ML_NATIVE)
        list(APPEND BASE_FLAGS -march=native)
    endif()
    set(AVX2_FLAGS   -mavx2 -mfma -mf16c)
    set(AVX512_FLAGS -mavx2 -mfma -mf16c -mavx512f -mavx512dq -mavx512bw -mavx512vl)
    set(AVX512BF16_FLAGS ${AVX512_FLAGS} -mavx512bf16)
    set(AVXVNNI_FLAGS    ${AVX2_FLAGS} -mavxvnni)
    set(AVX512VNNI_FLAGS ${AVX512_FLAGS} -mavx512vnni)
    set(SIMD_MATH_FLAGS -ffast-math -fno-finite-math-only)
elseif(MSVC)
    set(BASE_FLAGS /O2 /DNDEBUG)
    set(AVX2_FLAGS /arch:AVX2)
    set(AVX512_FLAGS /arch:AVX512)
    set(AVX512BF16_FLAGS /arch:AVX512)
    set(AVXVNNI_FLAGS    /arch:AVX2)
    set(AVX512VNNI_FLAGS /arch:AVX512)
    set(SIMD_MATH_FLAGS /fp:fast)
endif()

# ─── Sanitizers ───────────────────────────────────────────────────────────────
# Applied to every target defined below, so the kernels are instrumented, not
# only the test code. UBSan findings abort instead of printing and continuing.
if(SIMD_ML_SANITIZE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=undefined
                            -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
        message(STATUS "ASan + UBSan enabled for all targets")
    else()
        message(WARNING "SIMD_ML_SANITIZE=ON is only supported with GCC/Clang; ignored")
    endif()
endif()

# ─── ISA units ────────────────────────────────────────────────────────────────
# One object library per instruction set; kernels_<isa>.cpp in each defines
# that ISA's kernel table. dispatch.cpp learns which tables exist from
# SIMD_ML_HAVE_<ISA>.
set(SIMD_ML_SCALAR_SRCS
    src/dispatch/dispatch.cpp
    src/kernels/scalar_kernels.cpp
)
set(SIMD_ML_AVX2_SRCS
    src/kernels/kernels_avx2.cpp
    src/kernels/gemm/avx2_gemm_packed.cpp
    src/kernels/intrinsic_gelu.cpp
    src/kernels/activations/relu_avx2.cpp
    src/kernels/activations/silu_avx2.cpp
    src/kernels/activations/softmax_avx2.cpp
    src/kernels/activations/layer_norm_avx2.cpp
)
set(SIMD_ML_AVX512_SRCS
    src/kernels/kernels_avx512.cpp
    src/kernels/gemm/avx512_gemm_packed.cpp
)

# Extension units (no kernel table of their own), each called from a table
# unit after a CPUID check, and built only if the compiler accepts its flag:
#   avx512bf16  vdpbf16ps BF16 GEMM, from the AVX-512 table (GCC 10+, Clang 9+)
#   avxvnni     AVX-VNNI INT8 GEMM, from the AVX2 table (GCC 11+, Clang 12+)
#   avx512vnni  AVX512_VNNI INT8 GEMM, from the AVX-512 table (GCC 8+, Clang 6+)
# The calling unit gets SIMD_ML_HAVE_<UNIT>.
set(SIMD_ML_AVX512BF16_SRCS src/kernels/gemm/avx512bf16_gemm.cpp)
set(SIMD_ML_AVXVNNI_SRCS    src/kernels/gemm/avxvnni_gemm.cpp)
set(SIMD_ML_AVX512VNNI_SRCS src/kernels/gemm/avx512vnni_gemm.cpp)
set(_ext_units  avx512bf16   avxvnni   avx512vnni)
set(_ext_flags  -mavx512bf16 -mavxvnni -mavx512vnni)
set(_ext_parent avx512       avx2      avx512)

set(SIMD_ML_ISAS scalar)
if(SIMD_ML_ENABLE_X86)
    list(APPEND SIMD_ML_ISAS avx2 avx512)
    include(CheckCXXCompilerFlag)
    foreach(ext flag IN ZIP_LISTS _ext_units _ext_flags)
        if(MSVC)
            list(APPEND SIMD_ML_ISAS ${ext})
        else()
            string(TOUPPER ${ext} EXT)
            check_cxx_compiler_flag(${flag} SIMD_ML_CXX_HAS_${EXT})
            if(SIMD_ML_CXX_HAS_${EXT})
                list(APPEND SIMD_ML_ISAS ${ext})
            endif()
        endif()
    endforeach()
endif()
set(_isa_flags_scalar "")
set(_isa_flags_avx2   ${AVX2_FLAGS} ${SIMD_MATH_FLAGS})
set(_isa_flags_avx512 ${AVX512_FLAGS} ${SIMD_MATH_FLAGS})
set(_isa_flags_avx512bf16 ${AVX512BF16_FLAGS} ${SIMD_MATH_FLAGS})
set(_isa_flags_avxvnni    ${AVXVNNI_FLAGS} ${SIMD_MATH_FLAGS})
set(_isa_flags_avx512vnni ${AVX512VNNI_FLAGS} ${SIMD_MATH_FLAGS})
set(SIMD_ML_ISA_OBJECTS "")
foreach(isa IN LISTS SIMD_ML_ISAS)
    string(TOUPPER ${isa} ISA)
    add_library(simd_ml_isa_${isa} OBJECT ${SIMD_ML_${ISA}_SRCS})
    set_target_properties(simd_ml_isa_${isa} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(simd_ml_isa_${isa} PRIVATE src src/kernels)
    target_compile_options(simd_ml_isa_${isa} PRIVATE ${BASE_FLAGS} ${_isa_flags_${isa}})
    # Recorded in each kernel table; build_info() reports it.
    string(JOIN " " _flags_str ${BASE_FLAGS} ${_isa_flags_${isa}})
    target_compile_definitions(simd_ml_isa_${isa} PRIVATE "SIMD_ML_UNIT_FLAGS=\"${_flags_str}\"")
    target_compile_definitions(simd_ml_isa_scalar PRIVATE SIMD_ML_HAVE_${ISA})
    if(OpenMP_CXX_FOUND AND SIMD_ML_OPENMP)
        target_link_libraries(simd_ml_isa_${isa} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(simd_ml_isa_${isa} PRIVATE SIMD_ML_OPENMP)
    endif()
    list(APPEND SIMD_ML_ISA_OBJECTS $<TARGET_OBJECTS:simd_ml_isa_${isa}>)
    message(STATUS "ISA unit ${isa}: ${_flags_str}")
endforeach()
foreach(ext parent IN ZIP_LISTS _ext_units _ext_parent)
    if(ext IN_LIST SIMD_ML_ISAS)
        string(TOUPPER ${ext} EXT)
        target_compile_definitions(simd_ml_isa_${parent} PRIVATE SIMD_ML_HAVE_${EXT})
    endif()
endforeach()

# ─── Kernel library (shared between Python extension, tests and benchmarks) ───
add_library(simd_ml_kernels_lib STATIC ${SIMD_ML_ISA_OBJECTS})
# BUILD_INTERFACE: in-source build uses src/ and src/kernels/ directly.
# INSTALL_INTERFACE: installed consumers use ${prefix}/include/ so that
# #include <simd_ml_kernels/foo.hpp> resolves to ${prefix}/include/simd_ml_kernels/foo.hpp
target_include_directories(simd_ml_kernels_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/kernels>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
set_target_properties(simd_ml_kernels_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CXX)
if(OpenMP_CXX_FOUND AND SIMD_ML_OPENMP)
    message(STATUS "OpenMP C++ found; enabling multithreaded GEMM support")
    target_link_libraries(simd_ml_kernels_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(simd_ml_kernels_lib PUBLIC SIMD_ML_OPENMP)
else()
    message(STATUS "OpenMP support disabled or not found")
endif()

# Alias: simd_ml_kernels::simd_ml_kernels → simd_ml_kernels_lib
# This is the canonical target name that downstream find_package() consumers link against.
# The _lib suffix is an internal CMake convention; the public API name has no suffix.
add_library(simd_ml_kernels::simd_ml_kernels ALIAS simd_ml_kernels_lib)

# ─── Python extension module (optional) ───────────────────────────────────────
if(pybind11_FOUND)
    # The compiled module is simd_ml_kernels._core; the package's __init__.py
    # (python/simd_ml_kernels/) re-exports its public names.
    pybind11_add_module(simd_ml_kernels_core
        src/bindings/pybind_entry.cpp
    )
    set_target_properties(simd_ml_kernels_core PROPERTIES OUTPUT_NAME _core)

    # Build metadata reported by build_info() and `python -m simd_ml_kernels
    # info`, and recorded in benchmark JSON. The commit is taken when CMake
    # configures, so a build directory reused across commits reports the
    # commit it was configured at; "-dirty" marks uncommitted changes.
    execute_process(
        COMMAND git describe --always --dirty --abbrev=12
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE _simd_ml_git OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _simd_ml_git_rc ERROR_QUIET)
    if(NOT _simd_ml_git_rc EQUAL 0 OR NOT _simd_ml_git)
        set(_simd_ml_git "unknown")
    endif()
    target_compile_definitions(simd_ml_kernels_core PRIVATE
        SIMD_ML_VERSION="${PROJECT_VERSION}${SIMD_ML_PRERELEASE}"
        SIMD_ML_GIT_COMMIT="${_simd_ml_git}"
        SIMD_ML_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        # private bindings that call AVX2-unit functions directly exist only
        # when that unit is built (the macOS / scalar-only build has none)
        SIMD_ML_X86_AVAILABLE=$<BOOL:${SIMD_ML_ENABLE_X86}>)
    target_link_libraries(simd_ml_kernels_core PRIVATE simd_ml_kernels_lib)
    target_compile_options(simd_ml_kernels_core PRIVATE ${BASE_FLAGS})

    # CRITICAL: scikit-build-core builds a pip wheel by running `cmake --install`
    # and packaging whatever ends up installed. Once this project added other
    # install() rules below (static library, headers, CMake config, bench
    # executables) for `cmake --install` / find_package() consumers, the wheel
    # stopped containing the actual importable .so — because there was no
    # install() rule for the extension module itself. That produced wheels
    # that installed successfully but failed to import with
    # ModuleNotFoundError. The module is installed inside the package
    # directory as simd_ml_kernels/_core.
    install(TARGETS simd_ml_kernels_core
        LIBRARY DESTINATION simd_ml_kernels COMPONENT PythonModule
        RUNTIME DESTINATION simd_ml_kernels COMPONENT PythonModule
    )
else()
    message(STATUS "pybind11 not found: skipping Python extension build")
endif()

# ─── Benchmarks (x86 only) ───────────────────────────────────────────────────
# main_bench.cpp reads the TSC with x86 intrinsics and both harnesses call the
# AVX2 kernels directly, so they are AVX2 programs and run only on AVX2 CPUs.
if(SIMD_ML_ENABLE_X86 AND SIMD_ML_BUILD_BENCHMARKS)
    # ─── Standalone benchmark executable ─────────────────────────────────────────
    add_executable(bench
        src/main_bench.cpp
    )
    target_link_libraries(bench PRIVATE simd_ml_kernels_lib)
    target_include_directories(bench PRIVATE src)
    target_compile_options(bench PRIVATE ${BASE_FLAGS} ${AVX2_FLAGS})

    # ─── Statistical benchmark tool (extended harness) ──────────────────────────
    add_executable(bench_stat
        src/bench_stat.cpp
    )
    target_link_libraries(bench_stat PRIVATE simd_ml_kernels_lib)
    target_include_directories(bench_stat PRIVATE src)
    target_compile_options(bench_stat PRIVATE ${BASE_FLAGS} ${AVX2_FLAGS})

    # Optional OpenBLAS linkage for baseline comparisons.
    # Applied to BOTH bench and bench_stat: bench_stat's JSON output is what
    # is used for kernel development (published numbers come from
    # `python -m simd_ml_kernels bench`), so it needs
    # the same comparison capability as the RDTSC harness.
    if(BENCH_OPENBLAS)
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            target_link_libraries(bench PRIVATE ${BLAS_LIBRARIES})
            target_compile_definitions(bench PRIVATE BENCH_WITH_OPENBLAS)
            target_link_libraries(bench_stat PRIVATE ${BLAS_LIBRARIES})
            target_compile_definitions(bench_stat PRIVATE BENCH_WITH_OPENBLAS)
            message(STATUS "OpenBLAS found; baseline comparison enabled in bench and bench_stat")
        else()
            message(WARNING "BENCH_OPENBLAS=ON but BLAS not found; disabling")
        endif()
    endif()
endif()

# ─── Tests ───────────────────────────────────────────────────────────────────
if(SIMD_ML_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── Install rules ────────────────────────────────────────────────────────────
# Enables:  cmake --install <build_dir> --prefix /usr/local
# Installs: static library, all public headers, bench/bench_stat, CMake package config.
# Python extension is installed via pip (scikit-build-core), not cmake --install.
#
# After installation, downstream projects can use:
#   find_package(simd_ml_kernels REQUIRED)
#   target_link_libraries(my_target PRIVATE simd_ml_kernels::simd_ml_kernels)

# ── Static library ───────────────────────────────────────────────────────────
# Exclude the Python extension from install (pip installs it).
# Export the target so find_package() consumers can link against it.
install(TARGETS simd_ml_kernels_lib
    EXPORT  simd_ml_kernels-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    COMPONENT Development
)

# ── Public headers ───────────────────────────────────────────────────────────
# Install kernel headers (gemm/avx2_gemm_packed.hpp, intrinsic_gelu.hpp, etc.)
# Headers install into ${prefix}/include/simd_ml_kernels/ so that consumers can
# write:  #include <simd_ml_kernels/gemm/avx2_gemm_packed.hpp>
# The INSTALL_INTERFACE include path is ${CMAKE_INSTALL_INCLUDEDIR} (see above),
# so the compiler looks for simd_ml_kernels/... relative to that prefix — which is
# exactly where the install(DIRECTORY) puts the files.
install(DIRECTORY src/kernels/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/simd_ml_kernels"
    COMPONENT Development
    FILES_MATCHING PATTERN "*.hpp"
    PATTERN "gemm/gemm_dispatcher*" EXCLUDE
    PATTERN "gemm/naive_gemm*"      EXCLUDE
)
install(DIRECTORY src/dispatch/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/simd_ml_kernels"
    COMPONENT Development
    FILES_MATCHING PATTERN "*.hpp"
)

# ── Benchmark executables ─────────────────────────────────────────────────────
# Install under libexec/simd_ml_kernels so they don't clutter PATH.
if(SIMD_ML_ENABLE_X86 AND SIMD_ML_BUILD_BENCHMARKS)
    install(TARGETS bench bench_stat
        RUNTIME DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/simd_ml_kernels"
        COMPONENT Benchmarks
    )
endif()

# ── CMake package config ──────────────────────────────────────────────────────
# Generates simd_ml_kernels-config.cmake and simd_ml_kernels-config-version.cmake
# so downstream CMake projects can use find_package(simd_ml_kernels).
include(CMakePackageConfigHelpers)

set(SIMD_ML_KERNELS_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/simd_ml_kernels"
    CACHE PATH "Directory for simd_ml_kernels CMake package config files")

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/simd_ml_kernels-config-version.cmake"
    VERSION     "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/simd_ml_kernels-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/simd_ml_kernels-config.cmake"
    INSTALL_DESTINATION "${SIMD_ML_KERNELS_INSTALL_CMAKEDIR}"
)

# Install the exported target definition (needed for find_package to work)
install(EXPORT simd_ml_kernels-targets
    FILE        simd_ml_kernels-targets.cmake
    NAMESPACE   simd_ml_kernels::
    DESTINATION "${SIMD_ML_KERNELS_INSTALL_CMAKEDIR}"
    COMPONENT   Development
)

# Install the config files themselves
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/simd_ml_kernels-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/simd_ml_kernels-config-version.cmake"
    DESTINATION "${SIMD_ML_KERNELS_INSTALL_CMAKEDIR}"
    COMPONENT   Development
)

