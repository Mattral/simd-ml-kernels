# Changelog

All notable user-visible changes. The project follows semantic versioning;
before 1.0, minor versions may contain breaking changes, which are listed
under **Changed** with a migration note.

## Unreleased

### Fixed
- `build_info()["git_commit"]` is the bare 12-digit commit hash again (plus
  `-dirty` for local changes). Since the v0.3.0 tag, CMake's `git describe`
  had returned `v0.3.0-<n>-g<hash>`; benchmark JSONs recorded that form (one
  file, `lnl-226v-wsl_2026-10-19_290b91baad26.json`, keeps it as recorded).

### Added
- **`quantize_q4_0(W)`** and **`gemv_q4_0(x, Wq, N, K)`**: decode GEMV on
  weights in ggml's Q4_0 format (`block_q4_0`, byte-identical: float16
  scale + 16 bytes of 4-bit values; quantization as `quantize_row_q4_0_ref`,
  checked against blocks written by llama.cpp v0.7.0). x is quantized to Q8
  blocks once per call, as for `gemv_q8_0`; exact integer block dot
  products, two weight rows per register; same paths as
  `build_info()['q8_0_path']`. K % 32 != 0 raises `ArgumentError`.
  docs/DESIGN.md §2.9.
- `python -m simd_ml_kernels bench`: Q4_0 GEMV cases (`--cases q4_0`) against
  our Q8_0 and FP32 GEMV, PyTorch's FP32 `F.linear` on the dequantized
  weights and ggml's Q4_0 MUL_MAT on one thread.
- **`quantize_q8_0(W)`** and **`gemv_q8_0(x, Wq, N, K)`**: decode GEMV on
  weights in ggml's Q8_0 format (`block_q8_0`, byte-identical: float16 scale
  + 32 int8 per block; quantization as `quantize_row_q8_0_ref`). x is
  quantized to Q8 blocks once per call (float32 scales); exact integer
  block dot products with `vpdpbusd` (AVX-VNNI, AVX512_VNNI) or
  `vpmaddubsw` (AVX2), FP32 sums; `build_info()['q8_0_path']`. K % 32 != 0
  raises `ArgumentError`. docs/DESIGN.md §2.8. Measured on the reference laptop's P-core
  (`lnl-226v-wsl_2026-10-19_290b91baad26`): 92–95% of the read roof at 1×4096×4096,
  1×4096×11008 and 1×2048×8192; 3.49–3.61× our FP32 GEMV, 1.77–2.05× our
  BF16 GEMV, and 1.46–1.82× ggml's Q8_0 MUL_MAT (llama.cpp v0.7.0, one
  thread, `test-backend-ops perf`).
- `python -m simd_ml_kernels bench`: Q8_0 GEMV cases against our FP32/BF16
  GEMV, PyTorch's FP32 `F.linear` on the dequantized weights and, when
  `SIMD_ML_GGML_TEST_BACKEND_OPS` names llama.cpp's `test-backend-ops`,
  ggml's Q8_0 MUL_MAT on one thread; `--cases` to run a subset (development).

### Changed
- `sgemm_u8s8` with one row of A and `trans_b=True` (x·Wᵀ for one token)
  reads B in place instead of packing it, on every x86 path (exact integer
  sums; same results within the documented bound). At 1×4096×4096 on the
  laptop: 1.36× PyTorch's quantized linear, was 0.46× (`lnl-226v-wsl_2026-10-19_290b91baad26`).
- `bench`: each GEMV case measures its own read roof (the read stream timed
  interleaved with the case); `pct_stream_read` is relative to it
  (`roof_read_gbs`). A case with an implementation above 105% of its own roof
  is marked `roof_unstable` and not shown in the tables. The triad stays once
  per run.
- docs/ROADMAP.md: milestones renamed from "v0.2 … v1.0" to "Milestone 1 … 9".
  They were work stages, not package versions, and their names collided with
  real versions (the old "v0.3" milestone is not package 0.3.0).

## 0.3.0 — 2026-10-19

First release on PyPI (`pip install simd-ml-kernels`); 0.3.0rc1 was on
TestPyPI only. Changes since 0.2.0:

### Added
- `sgemm_f16(..., *, trans_b=False)` and `sgemm_bf16(..., *, trans_b=False)`
  (keyword-only, as `sgemm`'s): B stored `[N, K]`, C = A·Bᵀ, so a weight
  matrix in `torch.nn.Linear`'s layout needs no transpose copy. Existing
  calls are unchanged.
- Decode GEMV: with `trans_b` and at most 8 rows of A, `sgemm`, `sgemm_f16`
  and `sgemm_bf16` stream the rows of B in place (FP16/BF16 converted in
  registers, no packing) on the scalar, AVX2 and AVX-512 tables. Measured on
  the reference laptop's P-core: FP16/BF16 at 90–99% of the single-core read
  bandwidth and 1.78–1.92× PyTorch's `F.linear`; FP32 at 97–99%, 1.03–1.05×
  PyTorch. AVX-512: correct under SDE, speed not measured (the only AVX-512
  machine, a cloud VM, gave 5.8–11 GB/s between runs). docs/DESIGN.md §2.7.
- `python -m simd_ml_kernels bench`: memory-bandwidth roofs (a read-only
  stream and the STREAM triad, arrays of at least 4× the last-level cache,
  single core; `machine.stream_read_gbs`, `machine.stream_triad_gbs`) and
  decode GEMV cases (1×4096×4096, 1×4096×11008, 1×2048×8192; FP32, FP16,
  BF16) reported in GB/s and % of both roofs, against PyTorch's `F.linear`
  and, for FP32, OpenBLAS sgemv.
- **`sgemm_u8s8(A, B, a_scale, a_zero_point, b_scales, C=None, *, trans_b=False)`**:
  INT8 GEMM, uint8 activations × int8 weights with one scale per output
  column, float32 output (PyTorch's quantized-linear arithmetic, not
  requantized). AVX-VNNI and AVX512_VNNI kernels (`vpdpbusd`, exact int32
  sums per K block), an FP32 conversion path for AVX2 CPUs without VNNI, and
  an exact scalar version; `build_info()['u8s8_path']` names the path.
  Measured with AVX-VNNI on the reference laptop: 2.39–3.68× the FP32 GEMM and
  1.66–2.04× PyTorch's `quantized.linear` at N = 128–2048; a single row times
  a 4096×4096 matrix loses to PyTorch (0.46×); the conversion path runs at
  the FP32 GEMM's speed. AVX512_VNNI: correct under SDE; on a 2-vCPU Emerald
  Rapids cloud VM 1.43–2.75× the FP32 GEMM but 0.66–0.70× PyTorch from
  N = 512 up (`cloud-xeon-emr-2vcpu_2026-10-18_9029f1566ddb`).
  docs/DESIGN.md §4.2.
- `sgemm(..., trans_a=False, trans_b=False)` (keyword-only): C = alpha·op(A)·op(B)
  + beta·C with op(X) = X or Xᵀ, as BLAS transa/transb. A stored `[K, M]` /
  B stored `[N, K]` are read in place: the transpose happens while packing
  (8×8 register transposes for B), or for `trans_b` with M ≤ 8 in a
  dot-product path (the decode GEMV), so `x @ W.T` with `W` as
  `torch.nn.Linear` stores it needs no copy. `examples/linear_trans_b.py`.
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
  `selftest`ed after a clean install, and an sdist; `publish.yml` uploads them
  to TestPyPI or PyPI with Trusted Publishing (manual, from `main`).
- CI checks that the default kernel table matches the runner's CPU flags
  (an AVX-512 runner must use the AVX-512 table).
- `selftest` and the Python tests check every kernel table the CPU can run;
  ctest `isa_units` checks the object files contain only their own ISA.

### Changed
- Documentation: README rewritten for readers outside ML systems (what the
  library is for and not for, install, a decode example, what
  each measurement means); README links are absolute so they work on PyPI;
  stale claims fixed across the docs and docstrings (AVX-512 results from the
  cloud VM, SiLU and LayerNorm methods, the BF16 `vdpbf16ps` opt-in, the
  package description).
- **`sgemm_f16` / `sgemm_bf16` run the FP32 GEMM paths.** FP16/BF16 inputs
  are converted to FP32 once while packing A and B (or, for small products,
  per 6-row strip of A, and B in registers or once), and the 6×16 / 12×32 FP32
  micro-kernels run unchanged. Before, B was converted on every use with one
  accumulator per output strip and no cache blocking. Measured on the reference
  laptop: 95–101% of the FP32 GEMM at N = 128–2048, 1.6–1.8× it for a single
  row times a 4096×4096 matrix (B is half the bytes); FP16 1.41–1.46× and BF16
  0.97–1.16× PyTorch's float16 / bfloat16 `matmul`; the single-row case loses
  to PyTorch (0.58–0.63×). docs/DESIGN.md §4, docs/BENCHMARKS.md. Same API and
  error bound and the same results on every CPU, AVX512_BF16 ones included.
- **`vdpbf16ps` BF16 path is opt-in** (`SIMD_ML_BF16_DOT=1`, AVX512_BF16
  CPUs, one thread). Measured on an Emerald Rapids VM it ran at 64–73% of our
  FP32 GEMM, slower than the conversion path, so it is no longer the default
  there. When enabled it reads BF16 subnormal inputs as zero and flushes
  subnormal results. docs/DESIGN.md §4.1.
- `bench`: FP16/BF16 GEMM cases (against our FP32 GEMM and PyTorch), and on
  Linux glibc malloc thresholds are fixed (`protocol.malloc`), so allocating
  outputs costs the same in every case (docs/BENCHMARKS.md, "Allocation").
- **Threaded FP32 GEMM (OpenMP builds, `-DSIMD_ML_OPENMP=ON`; still off by
  default).** All threads of a call share one packed B panel per column block
  and take chunks of rows (and, for small M, columns) with a dynamic
  schedule, each packing its own rows of A; before, only whole NC-wide column
  blocks were split, so any N <= NC (2720 on the reference laptop) ran on one
  thread. Results are bitwise the same for every thread count. Products below
  about 2 MFLOP per thread use fewer threads; calls inside an OpenMP parallel
  region use one. The default thread count is OpenMP's (`OMP_NUM_THREADS`,
  else one per logical CPU) instead of 1, and `set_num_threads` no longer
  changes the thread count of the host program's own OpenMP regions.
  docs/DESIGN.md §2.4. Scaling is not measured yet.
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
- `layer_norm` (AVX2 and AVX-512 tables): two-pass statistics with four
  float accumulators per pass, flushed to double every 1024 elements, and the
  mean pass shifted by the row's first element, instead of one dependent
  double-precision add chain per pass; single-pass Welford was measured and
  was not faster for rows up to 1024 elements. Error bound documented and
  tested: `1e-5 + 1e-4·|ref|` for rows with `|mean| ≤ 30·std`.
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
