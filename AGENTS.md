# AGENTS.md

How to build, test, measure and extend simd-ml-kernels. Read this before
changing code; it is the only file you need for the routine tasks below.
Background on each kernel is in [docs/DESIGN.md](docs/DESIGN.md).

## Commands

```bash
# Python package (what users get). Installs simd_ml_kernels with the _core extension.
pip install .                          # or: pip install -e . --no-build-isolation
python -m simd_ml_kernels selftest     # every kernel vs float64; exit 1 on failure (< 1 s)
python -m simd_ml_kernels info         # compiled flags, commit, runtime ISA, CPU features
python -m pytest tests                 # Python tests (PyTorch cross-checks run if torch is installed)
python -m mypy.stubtest simd_ml_kernels --allowlist tests/stubtest_allowlist.txt

# C++ library, tests and harnesses
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build && ctest --test-dir build --output-on-failure
cmake -S . -B build-asan -G Ninja -DSIMD_ML_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo  # ASan+UBSan, all targets

# Benchmark (publishable only with the protocol in docs/BENCHMARKS.md)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c 2 python -m simd_ml_kernels bench --json benchmarks/results/<label>_<date>_<commit>.json \
    --label <label> --core "<which core>"
python benchmarks/make_tables.py       # regenerate README/BENCHMARKS tables from the JSON
```

Pitfalls:
- `build_info()["git_commit"]` is taken when CMake configures. A reused build
  directory reports the commit it was configured at; `-dirty` marks local
  changes. Benchmark only from a clean, fresh configure.
- The tests import the *installed* package. Re-run `pip install .` after
  changing C++ code, or the tests run the old extension.
- CMake options: `SIMD_ML_OPENMP` (threaded GEMM), `BENCH_OPENBLAS` (link BLAS
  into `bench_stat`), `SIMD_ML_ENABLE_X86=OFF` (non-x86 scalar build).

## Repository map

```
src/kernels/gemm/avx2_gemm_packed.{hpp,cpp}  FP32 GEMM behind sgemm: small-matrix path, 5-loop packed GEMM,
                                             8x8 AVX2 and 8x16 AVX-512 micro-kernels
src/kernels/gemm/{f16,bf16}_gemm.*           FP16 / BF16-input GEMM, FP32 accumulate
src/kernels/gemm/gemm_dispatcher.cpp         CPUID + XGETBV feature detection, cache detection, ISA label
src/kernels/gemm/naive_gemm.hpp              scalar reference GEMM
src/kernels/avx_matmul.{hpp,cpp}             second FP32 GEMM (6x16 / 6x32), C++ only: tests and bench
src/kernels/intrinsic_gelu.*                 GeLU exact (erf) and tanh
src/kernels/activations/                     relu, silu, softmax, layer_norm (+ activations.hpp)
src/kernels/simd_math.hpp                    fast_exp_avx2, tanh_avx2
src/kernels/cache_alloc.hpp                  64-byte aligned allocation, prefetch helpers
src/dispatch/                                CpuFeatures, CacheLevel, kernel registry declarations
src/bindings/pybind_entry.cpp                every Python entry point: validation, GIL release, docstrings
src/main_bench.cpp, src/bench_stat.cpp       C++ harnesses (development only)
python/simd_ml_kernels/                      __init__.py (re-exports), _core.pyi (stubs), _cli.py (info/selftest/bench)
python/simd_kernels/                         deprecated import alias (removed in 0.5.0)
tests/test_*.cpp                             C++ tests: one binary, simd_tests; run_*() suites called from test_runner.cpp
tests/test_*.py                              float64 references, edge cases, CLI, stubs
benchmarks/results/*.json                    committed benchmark results; make_tables.py renders them
examples/                                    short runnable scripts, run in CI
```

## Rules

Evidence
1. No number goes into README or docs unless it comes from a committed JSON in
   `benchmarks/results/` produced by `python -m simd_ml_kernels bench`, or from
   a test that asserts it. Never type result values into a JSON, a table or a
   script. If something cannot be measured here, write "not measured".
2. Compare against the strongest baseline on the same machine in the same run:
   OpenBLAS/MKL for GEMM, PyTorch CPU for activations. If we lose, say so.
3. A result JSON records `build.git_commit`; that commit must exist on `main`
   after the PR merges (benchmark after merging, or merge with a merge commit).

Numerics
4. Every kernel has a test against a float64 reference with an explicit
   tolerance (`|err| <= atol + rtol*|ref|`, never a bare relative error near
   zero), covering sizes 0, 1, 7, 8, 9, primes, block sizes ±1, unaligned
   pointers, and NaN/±Inf/subnormals where meaningful.
5. Approximations are named in the API (`approximate="tanh"`, as PyTorch names
   it) and their tested error bound is in the docstring.
6. Kernel files are built with `-ffast-math`: test for NaN/Inf on bit patterns,
   not with `isnan`/`x != x`, which the compiler may delete.

Code and repository
7. One concern per PR, descriptive title. Never commit to `main`, never
   force-push. Merging is the owner's decision.
8. No new documentation files; edit DESIGN.md, BENCHMARKS.md, ROADMAP.md, this
   file. Comments explain why, with numbers (latency, ports, cache sizes).
9. Public API changes need a deprecation alias and a CHANGELOG entry. Ask the
   owner before changing a default numerical behaviour, renaming the package,
   deleting a public function, or publishing anything.
10. No marketing words ("blazing", "production-grade", "state-of-the-art").

## Adding a kernel

1. **Header**: declare it in the kernel's header (`activations/activations.hpp`
   for elementwise ops) with the formula and the error bound you will test.
2. **Code paths**: AVX2+FMA in a `*_avx2.cpp`; an AVX-512 path only if it is
   measured to be faster; a scalar path for the tail (or masked loads, as in
   `intrinsic_gelu.cpp`) and for builds without AVX2 (`#ifndef __AVX2__`).
   Reuse `simd_math.hpp`; use `make_aligned_array<T>()` for scratch buffers.
3. **Build**: add the `.cpp` to `KERNEL_SRCS` in `CMakeLists.txt`.
4. **Dispatch**: if the kernel has more than one ISA path, select it at run time
   from `CpuFeatures` (never from compile-time macros alone) and report the
   choice where `detected_isa()` / `build_info()` can show it.
5. **Binding**: a `py_<name>` function in `pybind_entry.cpp` using
   `validate_array` / `check_writeable`, `DTypeError` / `ArgumentError`, and
   `py::gil_scoped_release` around the kernel; a NumPy-style docstring.
6. **Python surface**: export it in `python/simd_ml_kernels/__init__.py`
   (`__all__` too) and add it to `_core.pyi` (stubtest and
   `tests/test_stubs.py` fail otherwise).
7. **Tests**: a C++ test in `tests/test_*.cpp` (registered in
   `tests/test_runner.cpp`), a Python test against float64 in
   `tests/test_precision.py`, argument errors in
   `tests/test_bindings_edge_cases.py`, and a check in `selftest`
   (`selftest_checks()` in `_cli.py`) with the same tolerance.
8. **Benchmark**: add a case with its baseline to `cmd_bench` in `_cli.py` and,
   if it needs a new table, to `benchmarks/make_tables.py`.
9. **Docs**: a row in the README kernel table, its entry in
   `docs/api/python_api.md`, a section in `docs/DESIGN.md`, a CHANGELOG line.
10. **Verify**: `pip install .`, `selftest`, `pytest tests`, `ctest`, the ASan
    build, and stubtest, all passing; no new compiler warnings.

## Results and their schema

`python -m simd_ml_kernels bench --json` writes schema `simd-ml-kernels-bench/1`:

```
schema, command, python, timestamp_utc
machine   label, cpu_model, core, pinned_cpus, caches[] (CPUID leaf 4), logical_cpus, os, note
build     version, git_commit, compiler, kernel_flags, isa_compiled, openmp
baselines numpy, blas {name, version}, torch
env       OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS
protocol  warmup, reps, statistic, interleaved, clock, min_ghz
results[] kernel, impl, shape, median_s, p25_s, p75_s, min_s, reps,
          core_ghz {median, min, max, samples, attempts, below_min_ghz},
          gflops + flop_per_cycle (GEMM) | gelem_per_s + elem_per_cycle (elementwise)
```

Files are named `benchmarks/results/<label>_<YYYY-MM-DD>_<commit>.json`, one
machine per file. `benchmarks/make_tables.py` renders every file into the
generated blocks of README.md and docs/BENCHMARKS.md; CI fails if they differ.
