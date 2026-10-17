# Changelog

All notable user-visible changes. The project follows semantic versioning;
before 1.0, minor versions may contain breaking changes, which are listed
under **Changed** with a migration note.

## Unreleased

### Changed
- **Renamed to simd-ml-kernels.** Repository and PyPI distribution:
  `simd-ml-kernels` (was `SIMD-Microkernels-for-ML-Workloads` / `simd-ml-microkernels`);
  import: `simd_ml_kernels` (was `simd_kernels`); the "IntrinsicML" name is
  retired (it remains the title of the Zenodo preprint).
  - Migration: replace `import simd_kernels` with `import simd_ml_kernels`.
    `import simd_kernels` still works, returns the same module and emits a
    `DeprecationWarning`; it will be removed in 0.5.0.
  - The compiled extension is now `simd_ml_kernels._core`; the package
    re-exports every public name, so `simd_ml_kernels.sgemm` etc. are unchanged.
  - CMake: `find_package(simd_kernels)` / `simd_kernels::simd_kernels` became
    `find_package(simd_ml_kernels)` / `simd_ml_kernels::simd_ml_kernels`, and
    installed headers moved from `include/simd_kernels/` to
    `include/simd_ml_kernels/`. The CMake package was never published, so no
    alias is provided.
- `f16c_available()` and `bf16_avx512bf16_available()` now also require the OS
  to have enabled the YMM / ZMM register state (XGETBV), not only the CPUID bit.
- Wheel builds compile only the extension (new CMake options
  `SIMD_ML_BUILD_TESTS`, `SIMD_ML_BUILD_BENCHMARKS`, default ON for CMake users).

### Fixed
- `layer_norm(x, beta=b)` with `gamma=None` ignored `beta` (#5).
- Builds with clang/MinGW on Windows: aligned allocation uses C++17 aligned
  `operator new`; the benchmark harness uses `std::chrono` (#8).
