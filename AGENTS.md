# AGENTS.md

How to build, test, measure and extend simd-ml-kernels; the only file needed
for the routine tasks below. Background: [docs/DESIGN.md](docs/DESIGN.md).

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
# Threaded decode GEMV (OpenMP build: SKBUILD_CMAKE_DEFINE=SIMD_ML_OPENMP=ON pip install .), its own run
taskset -c 0-3 python -m simd_ml_kernels bench --cases threads --threads 1,2,4 \
    --json benchmarks/results/<label>_<date>_<commit>_threads.json --label <label> --core "<which cores>"
```

Pitfalls:
- `build_info()["git_commit"]` is taken at CMake configure time (`-dirty`:
  local changes); benchmark only from a clean, fresh configure.
- The tests import the *installed* package: `pip install .` after C++ changes.
- ASan binaries can segfault before `main` on WSL2 (ASLR); run them under `setarch -R`.
- CMake options: `SIMD_ML_OPENMP` (threaded GEMM), `BENCH_OPENBLAS` (link BLAS
  into `bench_stat`), `SIMD_ML_NATIVE` (`-march=native` everywhere; local
  benchmarks only), `SIMD_ML_ENABLE_X86=OFF` (scalar-only build).
- `SIMD_ML_BF16_DOT=1` opts `sgemm_bf16` into the `vdpbf16ps` path (AVX512_BF16
  CPUs; off by default, slower where measured).
- `SIMD_ML_GGML_TEST_BACKEND_OPS=<llama.cpp>/build/bin/test-backend-ops` adds ggml's
  Q8_0 and Q4_0 MUL_MAT to `bench` (and the ggml byte-comparison tests); unset: "not measured".
- `SIMD_ML_ISA=scalar|avx2` caps the kernel table, e.g. to benchmark AVX2 on
  an AVX-512 machine. AVX-512 without AVX-512 hardware: run under Intel SDE,
  `sde64 -skx -- python -m simd_ml_kernels selftest` (or `-nhm` for no AVX).

## Repository map

```
src/dispatch/kernels.hpp, dispatch.cpp       KernelTable per ISA, CPUID + XGETBV, table selection, SIMD_ML_ISA,
                                             GEMM block sizes from CPUID leaf 4 (gemm_blocking)
src/kernels/scalar_kernels.cpp               scalar unit: every kernel in portable C++, the scalar table
src/kernels/kernels_{avx2,avx512}.cpp        the AVX2 / AVX-512 tables
src/kernels/gemm/avx2_gemm_packed.{hpp,cpp}  FP32/FP16/BF16 GEMM, AVX2: 6x16 kernel, direct (small) and 5-loop packed paths
src/kernels/gemm/avx512_gemm_packed.cpp      the same on AVX-512 (12x32 kernel); avx512bf16_gemm.cpp: vdpbf16ps, CPUID-gated
src/kernels/gemm/{f16,bf16}_gemm.hpp         FP16 / BF16-input GEMM declarations (inputs converted to FP32 while packing)
src/kernels/gemm/int8_gemm.hpp, *vnni_gemm.cpp  INT8 GEMM (uint8 x int8, per-column scales): AVX-VNNI / AVX512_VNNI units;
                                             int8_rows.hpp: its M = 1, trans_b row-streaming path
src/kernels/quant/q8_0*                      ggml Q8_0 quantization (scalar unit, no fast-math) and gemv_q8_0
                                             (q8_0_rows.hpp: the YMM loop; avx2 / avxvnni / avx512vnni units)
src/kernels/quant/q4_0*                      ggml Q4_0 quantization and gemv_q4_0 (q4_0_rows.hpp: two rows per
                                             YMM; x through the Q8_0 driver gemv_q8_0_run)
src/kernels/gemm/naive_gemm.hpp              scalar reference GEMM (tests)
src/kernels/intrinsic_gelu.*                 GeLU exact (erf) and tanh (AVX2)
src/kernels/activations/                     relu, silu, softmax, layer_norm (AVX2; + activations.hpp)
src/kernels/simd_math.hpp                    fast_exp_avx2, tanh_avx2
src/kernels/cache_alloc.hpp                  64-byte aligned allocation, prefetch helpers
src/bindings/pybind_entry.cpp                every Python entry point: validation, GIL release, docstrings
src/main_bench.cpp, src/bench_stat.cpp       C++ harnesses (development only)
python/simd_ml_kernels/                      __init__.py (re-exports), _core.pyi (stubs), _cli.py (info/selftest/bench)
python/simd_kernels/                         deprecated import alias (removed in 0.5.0)
tests/test_*.cpp                             C++ tests: one binary, simd_tests, each suite run on every usable table
tests/check_isa_units.py                     ctest isa_units: no out-of-ISA instruction or shared inline code in a unit
tests/test_*.py                              float64 references, edge cases, CLI, stubs
tests/data/                                  fixtures from other projects (q4_0: llama.cpp's own blocks)
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
2. **Code paths**: a scalar version in `scalar_kernels.cpp` (no intrinsics,
   IEEE special values handled); AVX2+FMA in a `*_avx2.cpp` (scalar tail or
   masked loads, as in `intrinsic_gelu.cpp`); an AVX-512 version only if it is
   measured to be faster. Reuse `simd_math.hpp`; `make_aligned_array<T>()` for
   scratch buffers. Never call an ISA unit's function from outside its unit
   except through a table.
3. **Build**: add each `.cpp` to `SIMD_ML_{SCALAR,AVX2,AVX512}_SRCS` in
   `CMakeLists.txt` (its unit decides its flags).
4. **Dispatch**: add a field to `KernelTable` (`dispatch/kernels.hpp`) and set
   it in all three tables (`scalar_kernels.cpp`, `kernels_avx2.cpp`,
   `kernels_avx512.cpp`; AVX-512 may point at the AVX2 function).
5. **Binding**: a `py_<name>` function in `pybind_entry.cpp` that calls
   `simd_ml::kernels().<field>`, using `validate_array` / `check_writeable`,
   `DTypeError` / `ArgumentError`, and `py::gil_scoped_release` around the
   kernel; a NumPy-style docstring.
6. **Python surface**: export it in `python/simd_ml_kernels/__init__.py`
   (`__all__` too) and add it to `_core.pyi` (stubtest and
   `tests/test_stubs.py` fail otherwise).
7. **Tests**: a C++ test in `tests/test_*.cpp` calling `T->field` inside
   `for_each_kernel_table` (registered in `tests/test_runner.cpp`), a Python
   test against float64 (pytest runs it once per usable table) in
   `tests/test_precision.py`, argument errors in
   `tests/test_bindings_edge_cases.py`, and a check in `selftest`
   (`selftest_checks()` in `_cli.py`) with the same tolerance.
8. **Benchmark**: add a case with its baseline to `cmd_bench` in `_cli.py` and,
   if it needs a new table, to `benchmarks/make_tables.py`.
9. **Docs**: a row in the README kernel table, its entry in
   `docs/api/python_api.md`, a section in `docs/DESIGN.md`, a CHANGELOG line.
10. **Verify**: `pip install .`, `selftest`, `pytest tests`, `ctest` (including
    `isa_units`), the ASan build, stubtest, and `selftest` under SDE `-nhm`
    and `-skx`, all passing; no new compiler warnings.

## Results and their schema

`python -m simd_ml_kernels bench --json` writes schema `simd-ml-kernels-bench/1`:

```
schema, command, python, timestamp_utc
machine   label, cpu_model, core, pinned_cpus, caches[] (CPUID leaf 4), logical_cpus, os, note,
          stream_{read,triad}_gbs, stream_{read,triad}_allcore_gbs (null: no OpenMP or one CPU)
build     version, git_commit, compiler, kernel_flags, isa_compiled, openmp,
          sgemm_tile, sgemm_blocking {kc, mc, nc}, u8s8_path, q8_0_path
baselines numpy, blas {name, version}, torch, ggml {tag, commit, cmake, system_info, ...} | "not measured (...)"
env       OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS, SIMD_ML_ISA, SIMD_ML_BF16_DOT
protocol  warmup, reps, statistic, interleaved, clock, min_ghz, malloc, stream, gemv, gemv_roof,
          threads {counts, method, llama_3_2_1b} (threaded runs)
results[] kernel, impl, shape, median_s, p25_s, p75_s, min_s, reps,
          core_ghz {median, min, max, samples, attempts, below_min_ghz},
          gflops + flop_per_cycle (GEMM; GEMV also bytes, gb_per_s, pct_stream_{read,triad},
          weight_copies, roof_read_gbs, roof_unstable; Q8_0/Q4_0: err_vs_fp32; ggml rows: time_s,
          runs, threads_max/timed, command; threaded runs: threads, llama_3_2_1b) | gelem_per_s + elem_per_cycle (elementwise) | gb_per_s (stream_*)
```

Files are named `benchmarks/results/<label>_<YYYY-MM-DD>_<commit>.json` (threaded
runs: `..._threads.json`, rendered in their own blocks), one
machine per file. `benchmarks/make_tables.py` renders every file into the
generated blocks of README.md and docs/BENCHMARKS.md; CI fails if they differ.
