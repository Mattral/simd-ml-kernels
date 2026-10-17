# Changelog

All notable user-visible changes. The project follows semantic versioning;
before 1.0, minor versions may contain breaking changes, which are listed
under **Changed** with a migration note.

## Unreleased

### Added
- `python -m simd_ml_kernels selftest`: checks every kernel against a NumPy
  float64 reference with the bounds the test suite asserts (GEMMs: elementwise
  `|C − AB| ≤ 2·K·u·(|A||B|)`, u = 2⁻²⁴), prints one line per kernel and exits 1
  on any failure. Takes about 0.2 s. Replaces `tests/smoke_test.py`.
- `python -m simd_ml_kernels info [--json]`: package and build (version, git
  commit, compiler, kernel flags, compiled ISA, OpenMP) and CPU (model, runtime
  ISA, AVX-512, F16C, AVX512-BF16).
- `build_info()` gains `version`, `git_commit`, `compiler`, `kernel_flags`.
- `examples/`: `mlp_block.py` (GEMM → GeLU → GEMM + residual → LayerNorm,
  checked against PyTorch) and `bench_to_json.py`; both run in CI.
- `AGENTS.md`: commands, repository map, rules, add-a-kernel checklist, result
  schema. The design notes (`docs/architecture.md`, `docs/system_design.md`,
  `docs/design/*.md`) are merged into `docs/DESIGN.md`, with their errors
  corrected (listed at the end of that file).
- CI checks `_core.pyi` against the module: `mypy.stubtest` for names,
  `tests/test_stubs.py` for parameter names and defaults.

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
- **`gelu` / `gelu_inplace` default to exact GeLU**, `0.5·x·(1 + erf(x/√2))`,
  matching `torch.nn.functional.gelu`; `approximate="tanh"` selects the tanh
  formula used before. Outputs of the default call change by up to 4.7e-4
  (the gap between the two formulas). Migration: pass `approximate="tanh"` to
  keep the old results. Both variants are within 2e-6 of float64.
- `f16c_available()` and `bf16_avx512bf16_available()` now also require the OS
  to have enabled the YMM / ZMM register state (XGETBV), not only the CPUID bit.
- **Argument errors have specific types.** Wrong dtype raises
  `simd_ml_kernels.DTypeError` (a `TypeError`); wrong shape, layout, axis or
  `isa` string raises `simd_ml_kernels.ArgumentError` (a `ValueError`). Both
  also subclass `RuntimeError`, the type raised before, so existing
  `except RuntimeError` handlers keep working; that base is removed in 0.5.0.
- **An output `C` is never converted.** `sgemm`, `sgemm_f16` and `sgemm_bf16`
  now reject a `C` that is not a float32, C-contiguous, writeable ndarray.
  Previously such a `C` was silently copied, the result was written into the
  copy and the caller's array was left unchanged.
- **`sgemm_bf16` accepts only BF16 inputs**: uint16 bit patterns or
  `ml_dtypes.bfloat16`. float16 arrays were previously accepted and their bits
  read as BF16, which gives wrong results; they now raise `DTypeError`.
- Wheel builds compile only the extension (new CMake options
  `SIMD_ML_BUILD_TESTS`, `SIMD_ML_BUILD_BENCHMARKS`, default ON for CMake users).

- **`softmax` of a row with no finite maximum returns NaN**, as
  `torch.softmax` does: every element -inf, any NaN, or any +inf. An all -inf
  row used to return the uniform distribution 1/N. Partial -inf masks are
  unchanged.
- `set_num_threads(n > 1)` in a build without OpenMP now emits a
  `RuntimeWarning` instead of being silently ignored; `build_info()["openmp"]`
  reports whether OpenMP is compiled in.

### Removed
- Stray `gitignore` (duplicate of `.gitignore`) and `poetry.lock` (the build
  backend is scikit-build-core; nothing used it).

### Fixed
- `gelu(-inf)` returned NaN; it now returns -0 (x < -10 returns -0). The last
  `n % 8` elements of `gelu` were computed with libm `tanhf` instead of the
  vector kernel, so a value's result depended on its position; the tail now
  uses the same kernel with masked loads/stores. `gelu_inplace` called a
  function whose arguments were declared `__restrict__` with aliased pointers
  (undefined behaviour); the qualifier is removed.
- The `gelu_inplace` docstring claimed max error < 5e-6 vs exact GeLU; the
  tanh formula differs from exact GeLU by up to 4.7e-4.
- GEMM with K = 0 (`sgemm`, `sgemm_f16`, `sgemm_bf16`, and the scalar fallback)
  returned without touching C: an allocated output held uninitialised memory
  and `beta` was ignored. It now computes C = beta·C, with beta = 0
  overwriting C (NaN included), as BLAS does.
- `layer_norm` on an array whose last axis has length 0 crashed the
  interpreter (integer division by zero); it now returns an empty array.
- `layer_norm(x, beta=b)` with `gamma=None` ignored `beta` (#5).
- Builds with clang/MinGW on Windows: aligned allocation uses C++17 aligned
  `operator new`; the benchmark harness uses `std::chrono` (#8).
