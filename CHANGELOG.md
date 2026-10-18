# Changelog

All notable user-visible changes. The project follows semantic versioning;
before 1.0, minor versions may contain breaking changes, which are listed
under **Changed** with a migration note.

## Unreleased

### Added
- `sgemm(..., trans_a=False, trans_b=False)` (keyword-only): C = alpha·op(A)·op(B)
  + beta·C with op(X) = X or Xᵀ, as BLAS transa/transb. A stored `[K, M]` /
  B stored `[N, K]` are read in place: the transpose happens while packing
  (8×8 register transposes for B), or for `trans_b` with M ≤ 16 in a
  dot-product path, so `x @ W.T` with `W` as `torch.nn.Linear` stores it
  needs no copy. `examples/linear_trans_b.py`.
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
- **Portable builds with runtime dispatch.** Each instruction set has its own
  translation units compiled with explicit flags (scalar baseline;
  `-mavx2 -mfma -mf16c`; `-mavx512f -mavx512dq -mavx512bw -mavx512vl`)
  instead of compiling everything with `-march=native`. One kernel table per
  ISA is chosen at first use from CPUID + XGETBV, so a build runs on any
  x86-64 CPU (and, scalar only, on other architectures). Every kernel has a
  scalar version. `-DSIMD_ML_NATIVE=ON` restores `-march=native` for local
  benchmarking.
- `SIMD_ML_ISA=scalar|avx2|avx512` (environment) caps the kernel table.
- `build_info()` gains `isa_usable`, `isa_flags` and `sgemm_tile`.
- The AVX2/AVX-512 units use `-ffast-math -fno-finite-math-only` (NaN/Inf keep
  IEEE semantics; measured as fast or faster than `-ffast-math`); a test checks
  that importing and running the kernels leaves FTZ/DAZ unchanged.
- Wheels with cibuildwheel (`pyproject.toml` `[tool.cibuildwheel]`,
  `.github/workflows/wheels.yml`): manylinux x86_64 and macOS arm64, each
  `selftest`ed after a clean install; built as CI artifacts, not published.
- CI checks that the default kernel table matches the runner's CPU flags
  (an AVX-512 runner must use the AVX-512 table).
- `selftest` and the Python tests check every kernel table the CPU can run;
  ctest `isa_units` checks the object files contain only their own ISA.

### Changed
- **FP32 GEMM, AVX-512:** a 12×32 micro-kernel with 24 ZMM accumulators
  replaces the 8×16 kernel (8 accumulators), with the AVX2 path's five loops,
  CPUID-leaf-4 blocking and per-thread buffers. Verified under Intel SDE
  (`-skx`, `-spr`) and on AVX-512 CI runners; its speed is measured only on
  AVX-512 hardware (not yet).
- C++: the `KernelTable::sgemm` entry and `sgemm_packed_avx2/avx512` take
  `trans_a, trans_b` first; `sgemm_packed()` keeps the untransposed signature.
- **FP32 GEMM, AVX2:** one 6×16 micro-kernel (12 YMM accumulators) replaces
  the 8×8 kernel and the one-accumulator small-matrix loop. Small products
  (up to 80³, N ≤ 16 or M = 1) run it on A and B in place; the rest go through
  the packed five loops. KC/MC/NC are derived at run time from CPUID leaf 4
  (`build_info()["sgemm_blocking"]`, also recorded in benchmark JSON) instead
  of fixed constants. Packing buffers are per thread and kept, instead of
  2.1 MiB allocated and freed on every call. docs/DESIGN.md §2 has the
  reasons and the measurements behind each choice.
- `build_info()` gains `sgemm_blocking` (`{kc, mc, nc}`); `info` and the
  benchmark JSON `build` block gain `sgemm_tile` and `sgemm_blocking`.
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
- `sgemm(..., isa="scalar")` runs the scalar kernel instead of raising
  "not implemented"; an `isa=` the build or CPU cannot run raises
  `RuntimeError` as before, and `isa="cuda"`-style typos raise `ArgumentError`.
- `sgemm_f16` runs on CPUs without F16C (scalar kernel) instead of raising.
- `build_info()["isa_compiled"]` lists the kernel tables in the build
  (`"scalar avx2 avx512"`) instead of one compile-time label;
  `kernel_flags` are the active table's flags; the `fma` key is removed.
  `detected_isa()` no longer returns `"sse42"` (such CPUs use `"scalar"`).
- C++: `kernel_registry.hpp` (`dispatch::get_kernels`, `dispatch::sgemm`,
  `dispatch::detected_isa`) is replaced by `dispatch/kernels.hpp`
  (`kernels()`, `kernels_for()`); `gemm::set_gemm_isa_override`,
  `get_gemm_isa_override`, `gemm_packed_isa_is_avx512`,
  `gemm_packed_avx512_hardware_available` and `activations::gelu_avx2` are
  removed. `gemm::sgemm_packed` still exists and dispatches.

### Removed
- `avx_matmul.cpp` / `avx_matmul_avx512.cpp` (`simd_sgemm`), the second FP32
  GEMM, which only the C++ tests and the `bench` harness could call. Its 6×16
  shape is now the AVX2 kernel of `sgemm`; `bench` measures `sgemm_packed`.
- Stray `gitignore` (duplicate of `.gitignore`) and `poetry.lock` (the build
  backend is scikit-build-core; nothing used it).

### Fixed
- `silu` (AVX2 and AVX-512 tables) computed `x·0.5·(1 + tanh(x/2))` with
  tanh clamped at ±7.99, so its error grew without bound below x ≈ −16
  (2e-6 at −20, 1e-4 at −1000, 5e31 at −FLT_MAX). It now computes
  `x / (1 + exp(−x))`: `|err| ≤ 1e-35 + 1e-6·|ref|` for every finite x,
  and 1.5× faster on the Lunar Lake P-core (L1-resident). `silu(-inf)` is
  now NaN on every table (was −inf on AVX2), as in PyTorch and the scalar
  kernel. The last `n % 8` elements use the vector kernel (masked), so a
  value's result no longer depends on its position.
- CMake detected x86-64 from `CMAKE_SYSTEM_PROCESSOR`, which is empty when
  `PROCESSOR_ARCHITECTURE` is not set (seen in an MSYS2 shell); the build then
  silently became scalar-only and the extension failed to link. The compiler's
  own target macros are used now.
- The C++ tests no longer download doctest at configure time: it was fetched
  but never used (`test_doctest.cpp` was not compiled; now removed).
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
